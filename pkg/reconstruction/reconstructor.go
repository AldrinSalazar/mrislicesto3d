package reconstruction

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"gonum.org/v1/gonum/stat"

	"mrislicesto3d/pkg/shearlet"
	"mrislicesto3d/pkg/stl"
)

// Params holds the reconstruction parameters as described in the paper.
// These parameters control the input/output and processing configuration.
type Params struct {
	// InputDir is the directory containing 2D MRI slice images in JPEG format.
	// Images should be sorted alphanumerically to maintain proper slice order.
	InputDir string

	// OutputFile is the path where the resulting 3D model will be saved in STL format.
	OutputFile string

	// NumCores specifies how many CPU cores to use for parallel processing.
	// The paper demonstrates significant speedup with multiple cores.
	NumCores int

	// SliceGap represents the physical distance between consecutive MRI slices in mm.
	// This affects the z-axis scaling of the final 3D model.
	SliceGap float64

	// SaveIntermediaryResults determines whether to save intermediary processing results.
	// When enabled, the algorithm will save images at various processing stages.
	SaveIntermediaryResults bool

	// IntermediaryDir is the directory where intermediary results will be saved.
	// Only used when SaveIntermediaryResults is true.
	IntermediaryDir string

	// Verbose controls the level of logging output.
	// When enabled, detailed progress and debug information will be displayed.
	Verbose bool

	// IsoLevelPercent controls the threshold for final volume generation in marching cubes.
	// Values range from 0.0 to 1.0, with lower values creating more inclusive models.
	// Default is 0.25 (25% of the range from min to max intensity).
	IsoLevelPercent float64

	// EdgeDetectionThreshold controls the sensitivity of edge detection.
	// Values range from 0.0 to 1.0, with lower values detecting more edges.
	// Default is 0.5.
	EdgeDetectionThreshold float64

	// ShearletScales is the number of scales for the shearlet transform.
	// Higher values provide more detailed edge detection but increase computation time.
	// Default is 3.
	ShearletScales int

	// ShearletShears is the number of shears for the shearlet transform.
	// Higher values provide more directional sensitivity but increase computation time.
	// Default is 8.
	ShearletShears int

	// ShearletConeParam is the cone parameter for the shearlet transform.
	// Controls the shape of the frequency domain partitioning.
	// Default is 1.0.
	ShearletConeParam float64
}

// Reconstructor handles the 3D reconstruction process following the methodology
// described in "Fast 3D Volumetric Image Reconstruction from 2D MRI Slices by
// Parallel Processing" by Somoballi Ghoshal et al.
//
// The reconstruction process consists of several steps:
// 1. Loading and preprocessing input slices
// 2. Applying Shearlet transform for denoising
// 3. Dividing dataset into quadrants for parallel processing
// 4. Processing sub-volumes with edge-preserved kriging interpolation
// 5. Merging sub-volumes and generating 3D mesh using marching cubes
// 6. Calculating quality metrics
type Reconstructor struct {
	// params stores the reconstruction configuration
	params *Params

	// slices holds the original MRI slice images
	slices []image.Image

	// subSlices contains the divided dataset organized as:
	// [subsetIdx][sliceIdx][quadrantIdx]
	// This division enables efficient parallel processing
	subSlices [][][]image.Image

	// width and height store the dimensions of the input slices
	width  int
	height int
}

// NewReconstructor creates a new reconstructor instance with the provided parameters.
// This is the entry point for starting the reconstruction process.
//
// Parameters:
//   - params: Configuration parameters for the reconstruction process
//
// Returns:
//   - A new Reconstructor instance initialized with the provided parameters
func NewReconstructor(params *Params) *Reconstructor {
	return &Reconstructor{
		params: params,
		slices: make([]image.Image, 0),
	}
}

// Process runs the complete reconstruction pipeline
func (r *Reconstructor) Process() error {
	// Step 1: Load input slices
	fmt.Println("Step 1: Loading input slices...")
	if err := r.loadSlices(); err != nil {
		return fmt.Errorf("failed to load slices: %v", err)
	}

	// Save original slices as intermediary result
	if r.params.SaveIntermediaryResults {
		if r.params.Verbose {
			fmt.Println("Saving original slices...")
		}
		for i, slice := range r.slices {
			if err := r.saveIntermediaryResult("01_original_slices", slice, i); err != nil {
				if r.params.Verbose {
					fmt.Printf("Warning: Failed to save original slice %d: %v\n", i, err)
				}
			}
		}
	}

	// Step 2: Apply shearlet transform for denoising
	fmt.Println("Step 2: Applying shearlet transform for denoising...")
	if err := r.denoiseSlices(); err != nil {
		return fmt.Errorf("failed to denoise slices: %v", err)
	}

	// Save denoised slices as intermediary result
	if r.params.SaveIntermediaryResults {
		if r.params.Verbose {
			fmt.Println("Saving denoised slices...")
		}
		for i, slice := range r.slices {
			if err := r.saveIntermediaryResult("02_denoised_slices", slice, i); err != nil {
				if r.params.Verbose {
					fmt.Printf("Warning: Failed to save denoised slice %d: %v\n", i, err)
				}
			}
		}
	}

	// Step 3: Divide dataset for parallel processing
	fmt.Println("Step 3: Dividing dataset for parallel processing...")
	if err := r.divideDataset(); err != nil {
		return fmt.Errorf("failed to divide dataset: %v", err)
	}

	// Save divided dataset as intermediary result
	if r.params.SaveIntermediaryResults {
		if r.params.Verbose {
			fmt.Println("Saving divided dataset...")
		}
		for s := 0; s < len(r.subSlices); s++ {
			for i := 0; i < len(r.subSlices[s]); i++ {
				for q := 0; q < len(r.subSlices[s][i]); q++ {
					if err := r.saveIntermediaryResult(fmt.Sprintf("03_divided_dataset/subset_%d", s), r.subSlices[s][i][q], i*4+q); err != nil {
						if r.params.Verbose {
							fmt.Printf("Warning: Failed to save quadrant %d of slice %d in subset %d: %v\n",
								q, i, s, err)
						}
					}
				}
			}
		}
	}

	// Step 4: Process sub-volumes with edge-preserved kriging
	fmt.Println("Step 4: Processing sub-volumes with edge-preserved kriging...")
	subVolumes, err := r.processSubVolumesInParallel()
	if err != nil {
		return fmt.Errorf("failed to process sub-volumes: %v", err)
	}

	// Save processed sub-volumes as intermediary result
	if r.params.SaveIntermediaryResults {
		if r.params.Verbose {
			fmt.Println("Saving processed sub-volumes...")
		}
		for i, subVolume := range subVolumes {
			if err := r.saveIntermediaryResult("04_processed_subvolumes", subVolume, i); err != nil {
				if r.params.Verbose {
					fmt.Printf("Warning: Failed to save sub-volume %d: %v\n", i, err)
				}
			}
		}
	}

	// Step 5: Merge sub-volumes and generate 3D mesh
	fmt.Println("Step 5: Merging sub-volumes and generating 3D mesh...")
	if err := r.mergeAndGenerateSTL(subVolumes); err != nil {
		return fmt.Errorf("failed to merge and generate STL: %v", err)
	}

	// Save merged volume slices as intermediary result
	if r.params.SaveIntermediaryResults {
		if r.params.Verbose {
			fmt.Println("Saving merged volume slices...")
		}

		// Get the merged volume data
		volumeData, width, height, depth := r.GetVolumeData()

		// Save a few slices as examples
		sampleIndices := []int{0, depth / 2, depth - 1}
		for _, z := range sampleIndices {
			if z >= depth {
				continue
			}

			// Extract the slice
			slice := make([]float64, width*height)
			for y := 0; y < height; y++ {
				for x := 0; x < width; x++ {
					idx := z*width*height + y*width + x
					if idx < len(volumeData) {
						slice[y*width+x] = volumeData[idx]
					}
				}
			}

			if err := r.saveIntermediaryResult("05_merged_volume", slice, z); err != nil {
				if r.params.Verbose {
					fmt.Printf("Warning: Failed to save merged volume slice %d: %v\n", z, err)
				}
			}
		}
	}

	return nil
}

// loadSlices loads and sorts the input MRI slices from the specified directory.
// This function performs the following operations:
// 1. Reads all files from the input directory
// 2. Filters for JPEG image files which contain the MRI slice data
// 3. Sorts the files based on numerical values in their filenames to ensure anatomical order
// 4. Loads each slice as an image and stores it in the reconstructor
//
// The proper ordering of slices is critical for accurate 3D reconstruction,
// as it preserves the spatial relationship between adjacent anatomical structures.
//
// Returns:
//   - nil if successful, or an error if file reading or image loading fails
func (r *Reconstructor) loadSlices() error {
	// Get list of files in the input directory
	files, err := ioutil.ReadDir(r.params.InputDir)
	if err != nil {
		return fmt.Errorf("failed to read input directory: %v", err)
	}

	// Filter for image files and sort by name
	var imageFiles []string
	for _, file := range files {
		if !file.IsDir() && strings.HasSuffix(strings.ToLower(file.Name()), ".jpg") {
			imageFiles = append(imageFiles, filepath.Join(r.params.InputDir, file.Name()))
		}
	}

	// Sort files by the numeric part of their names
	sort.Slice(imageFiles, func(i, j int) bool {
		numI := extractNumber(imageFiles[i])
		numJ := extractNumber(imageFiles[j])
		return numI < numJ
	})

	// Load each image
	for _, file := range imageFiles {
		img, err := loadImage(file)
		if err != nil {
			return fmt.Errorf("failed to load image %s: %v", file, err)
		}
		r.slices = append(r.slices, img)
	}

	// Check if we have any slices
	if len(r.slices) == 0 {
		return fmt.Errorf("no valid image files found in input directory")
	}

	// Get dimensions from the first slice
	r.width = r.slices[0].Bounds().Dx()
	r.height = r.slices[0].Bounds().Dy()

	// Print information about loaded slices
	if r.params.Verbose {
		fmt.Printf("Loaded %d slices with dimensions %dx%d\n", len(r.slices), r.width, r.height)
		fmt.Printf("Inter-slice gap: %.1f mm\n", r.params.SliceGap)
	}

	return nil
}

// extractNumber extracts the numeric part from a filename
func extractNumber(filename string) int {
	// This is a simple implementation - adjust as needed for your naming convention
	base := filepath.Base(filename)
	numStr := ""
	for _, c := range base {
		if c >= '0' && c <= '9' {
			numStr += string(c)
		}
	}

	if numStr != "" {
		num, err := strconv.Atoi(numStr)
		if err == nil {
			return num
		}
	}
	return 0
}

// denoiseSlices applies the Shearlet transform for edge-preserving denoising
// as described in section 3.1 of the paper.
//
// The Shearlet transform is particularly well-suited for medical image processing
// because it can effectively capture directional features (like edges and contours)
// while still removing noise. This implementation uses the following steps:
//
//  1. For each MRI slice:
//     a. Convert the image to a float array for mathematical processing
//     b. Apply the edge-preserved smoothing algorithm using Shearlet coefficients
//     c. Convert the processed data back to an image format
//
// Unlike traditional denoising methods that might blur important anatomical boundaries,
// this approach preserves edge information that is critical for accurate 3D reconstruction.
//
// Returns:
//   - nil if successful (processing is done in-place on the slice collection)
func (r *Reconstructor) denoiseSlices() error {
	if r.params.Verbose {
		fmt.Println("Applying shearlet transform for denoising...")
	}

	// Create a shearlet transform instance with configurable parameters
	scales := r.params.ShearletScales
	shears := r.params.ShearletShears
	coneParam := r.params.ShearletConeParam

	// Use default values if not specified
	if scales <= 0 {
		scales = 3
	}
	if shears <= 0 {
		shears = 8
	}
	if coneParam <= 0 {
		coneParam = 1.0
	}

	transform := shearlet.NewTransform(scales, shears, coneParam)

	// Process each slice
	for i, slice := range r.slices {
		// Convert image to float array
		data := imageToFloat(slice)

		// Apply shearlet transform for denoising
		denoised := transform.ApplyEdgePreservedSmoothing(data)

		// Convert back to image
		r.slices[i] = floatToImage(denoised, r.width, r.height)
	}

	return nil
}

// divideDataset divides the MRI slices into 8 equal parts (4 quadrants x 2 subsets)
// as described in section 4.1.1 of the paper. This division is a critical component
// of the parallel processing approach that enables significant performance improvements.
//
// The division process follows these steps:
// 1. Divide slices into 2 subsets along the z-axis (anatomical axis)
// 2. For each slice, divide it into 4 quadrants (top-left, top-right, bottom-left, bottom-right)
//
// This creates 8 independent sub-volumes that can be processed in parallel.
// The division is designed to minimize data dependencies between sub-volumes
// while maintaining the ability to reconstruct the full volume accurately.
//
// Returns:
//   - nil if successful, or an error if there are insufficient slices
func (r *Reconstructor) divideDataset() error {
	numSlices := len(r.slices)
	if numSlices < 2 {
		return fmt.Errorf("insufficient slices for processing, need at least 2")
	}

	// Determine how many subsets to create (k=2 in the paper example)
	// The paper describes dividing the dataset into 2k subsets in the z direction,
	// where k is the number of available computing nodes
	numSubsets := 2

	// Each subset will have numSlices/numSubsets slices
	// Using ceiling division to ensure all slices are covered
	slicesPerSubset := (numSlices + numSubsets - 1) / numSubsets

	// Initialize the 3D array organized as:
	// [subsetIdx][sliceIdx][quadrantIdx]
	r.subSlices = make([][][]image.Image, numSubsets)

	// For each subset along the z-axis
	for i := 0; i < numSubsets; i++ {
		// Calculate start and end indices for this subset
		startIdx := i * slicesPerSubset
		endIdx := (i + 1) * slicesPerSubset

		// Handle the case where the last subset might have fewer slices
		if endIdx > numSlices {
			endIdx = numSlices
		}

		// Create a subset with its slices
		subset := make([][]image.Image, endIdx-startIdx)

		// For each slice in this subset
		for j := startIdx; j < endIdx; j++ {
			// Split the slice into 4 quadrants (spatial division)
			// This creates the divisions in the x-y plane
			quadrants := r.splitImageIntoQuadrants(r.slices[j])
			subset[j-startIdx] = quadrants
		}

		// Store the complete subset with its quadrants
		r.subSlices[i] = subset
	}

	fmt.Printf("Divided dataset into %d subsets of slices, each with 4 quadrants\n", numSubsets)
	return nil
}

// splitImageIntoQuadrants splits an image into 4 equal quadrants
// following Algorithm 1, Step 1 from the paper
func (r *Reconstructor) splitImageIntoQuadrants(img image.Image) []image.Image {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Calculate half dimensions
	halfWidth := width / 2
	halfHeight := height / 2

	// Create 4 quadrants
	quadrants := make([]image.Image, 4)

	// Create each quadrant
	for q := 0; q < 4; q++ {
		// Determine quadrant boundaries
		x0 := (q % 2) * halfWidth
		y0 := (q / 2) * halfHeight

		rect := image.Rect(0, 0, halfWidth, halfHeight)
		quadImg := image.NewRGBA(rect)

		// Copy pixels to the quadrant
		for y := 0; y < halfHeight; y++ {
			for x := 0; x < halfWidth; x++ {
				srcX := x0 + x
				srcY := y0 + y

				if srcX < width && srcY < height {
					quadImg.Set(x, y, img.At(srcX, srcY))
				}
			}
		}

		quadrants[q] = quadImg
	}

	return quadrants
}

// processSubVolumesInParallel processes all sub-volumes in parallel using multiple cores
// This implements Algorithm 1 from the paper for parallel processing
func (r *Reconstructor) processSubVolumesInParallel() ([][][]float64, error) {
	// Create a slice to hold all sub-volumes
	numSubsets := len(r.subSlices)
	numQuadrants := 4 // Each slice is divided into 4 quadrants

	if r.params.Verbose {
		fmt.Printf("Divided dataset into %d subsets of slices, each with 4 quadrants\n", numSubsets)
	}

	// Initialize the result array
	subVolumes := make([][][]float64, numSubsets)
	for i := range subVolumes {
		subVolumes[i] = make([][]float64, numQuadrants)
	}

	// Create a channel for results
	type processingResult struct {
		subsetIdx   int
		quadrantIdx int
		data        []float64
		err         error
	}
	resultChan := make(chan processingResult)

	// Count total tasks for progress reporting
	totalTasks := numSubsets * numQuadrants
	completedTasks := 0

	// Create a wait group to wait for all goroutines
	var wg sync.WaitGroup

	// Limit the number of concurrent goroutines
	semaphore := make(chan struct{}, r.params.NumCores)

	// Process each subset and quadrant in parallel
	for s := 0; s < numSubsets; s++ {
		for q := 0; q < numQuadrants; q++ {
			wg.Add(1)
			go func(subsetIdx, quadrantIdx int) {
				defer wg.Done()

				// Acquire semaphore
				semaphore <- struct{}{}
				defer func() { <-semaphore }()

				// Extract the slices for this quadrant
				var slices []image.Image
				for i := 0; i < len(r.subSlices[subsetIdx]); i++ {
					slices = append(slices, r.subSlices[subsetIdx][i][quadrantIdx])
				}

				// Process the sub-volume
				data, err := r.processSubVolume(slices)

				// Send the result
				resultChan <- processingResult{
					subsetIdx:   subsetIdx,
					quadrantIdx: quadrantIdx,
					data:        data,
					err:         err,
				}
			}(s, q)
		}
	}

	// Start a goroutine to close the result channel when all processing is done
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// Collect results and update progress
	for result := range resultChan {
		if result.err != nil {
			return nil, fmt.Errorf("failed to process sub-volume %d-%d: %v",
				result.subsetIdx, result.quadrantIdx, result.err)
		}

		// Store the result
		subVolumes[result.subsetIdx][result.quadrantIdx] = result.data

		// Update progress
		completedTasks++
		progress := float64(completedTasks) / float64(totalTasks) * 100.0
		if r.params.Verbose {
			fmt.Printf("\rProcessing sub-volumes: %.1f%% complete", progress)
		}
	}

	if r.params.Verbose {
		fmt.Println() // New line after progress
	}

	return subVolumes, nil
}

// processSubVolume processes a single sub-volume using edge-preserved kriging
// as described in Algorithm 2 of the paper
func (r *Reconstructor) processSubVolume(slices []image.Image) ([]float64, error) {
	// Convert slices to float arrays
	sliceData := make([][]float64, len(slices))
	for i, slice := range slices {
		sliceData[i] = imageToFloat(slice)
	}

	// Get dimensions of a single quadrant
	quadWidth := r.width / 2
	quadHeight := r.height / 2

	// Apply edge-preserved kriging interpolation
	if r.params.Verbose {
		fmt.Println("Applying edge-preserved kriging interpolation...")
	}

	// Apply Shearlet transform for edge detection
	if r.params.Verbose {
		fmt.Println("Applying Shearlet transform for edge detection...")
	}

	// Create a shearlet transform for edge detection with configurable parameters
	scales := r.params.ShearletScales
	shears := r.params.ShearletShears
	coneParam := r.params.ShearletConeParam

	// Use default values if not specified
	if scales <= 0 {
		scales = 3
	}
	if shears <= 0 {
		shears = 8
	}
	if coneParam <= 0 {
		coneParam = 1.0
	}

	transform := shearlet.NewTransform(scales, shears, coneParam)

	// Detect edges in each slice
	edgeInfo := make([]shearlet.EdgeInfo, len(slices))
	for i, data := range sliceData {
		edgeInfo[i] = transform.DetectEdgesWithOrientation(data)
	}

	// Apply edge-preserved smoothing
	if r.params.Verbose {
		fmt.Println("Applying edge-preserved smoothing...")
	}

	// Calculate the number of slices in the interpolated volume
	slicesPerGap := int(r.params.SliceGap)
	if slicesPerGap < 1 {
		slicesPerGap = 1
	}

	// Calculate total number of slices in the interpolated volume
	totalSlices := (len(slices)-1)*slicesPerGap + 1

	// Create the 3D volume
	volumeData := make([]float64, quadWidth*quadHeight*totalSlices)

	// Interpolate between slices
	for z := 0; z < totalSlices; z++ {
		// Calculate the corresponding position in the original slices
		origZ := float64(z) / float64(slicesPerGap)

		// Get the two nearest slices
		z1 := int(origZ)
		z2 := z1 + 1
		if z2 >= len(slices) {
			z2 = len(slices) - 1
		}

		// Calculate interpolation weight
		weight := origZ - float64(z1)

		// For each pixel in the slice
		for y := 0; y < quadHeight; y++ {
			for x := 0; x < quadWidth; x++ {
				// Get index in the current slice
				idx := y*quadWidth + x

				// Check if this is an edge pixel in either of the two slices
				isEdge := false
				if z1 < len(edgeInfo) && idx < len(edgeInfo[z1].Edges) && edgeInfo[z1].Edges[idx] > 0.5 {
					isEdge = true
				}
				if z2 < len(edgeInfo) && idx < len(edgeInfo[z2].Edges) && edgeInfo[z2].Edges[idx] > 0.5 {
					isEdge = true
				}

				// Apply different interpolation based on whether this is an edge
				var value float64
				if isEdge {
					// For edge pixels, use edge-preserved kriging
					// Since we don't have a direct InterpolateEdgePreserved method,
					// we'll use a weighted average that preserves edges better
					// This is a simplified version of what the paper describes
					edgeWeight1 := 0.5
					edgeWeight2 := 0.5

					if z1 < len(edgeInfo) && idx < len(edgeInfo[z1].Edges) {
						edgeWeight1 = math.Max(0.5, edgeInfo[z1].Edges[idx])
					}
					if z2 < len(edgeInfo) && idx < len(edgeInfo[z2].Edges) {
						edgeWeight2 = math.Max(0.5, edgeInfo[z2].Edges[idx])
					}

					// Normalize weights
					sum := edgeWeight1 + edgeWeight2
					edgeWeight1 /= sum
					edgeWeight2 /= sum

					// Apply weighted interpolation
					value = sliceData[z1][idx]*edgeWeight1*(1-weight) + sliceData[z2][idx]*edgeWeight2*weight
				} else {
					// For non-edge pixels, use simple linear interpolation
					value = sliceData[z1][idx]*(1-weight) + sliceData[z2][idx]*weight
				}

				// Store the interpolated value in the volume
				volumeData[z*quadWidth*quadHeight+y*quadWidth+x] = value
			}
		}
	}

	return volumeData, nil
}

// mergeAndGenerateSTL merges the sub-volumes and generates an STL file
// This implements the merging step from section 4.2 of the paper
func (r *Reconstructor) mergeAndGenerateSTL(subVolumes [][][]float64) error {
	// Calculate dimensions of each sub-volume and the full volume
	numSubsets := len(subVolumes)
	numQuadrants := len(subVolumes[0])

	// Determine the dimensions of a single quadrant
	// Each quadrant is 1/4 of the original slice (half width, half height)
	quadWidth := r.width / 2
	quadHeight := r.height / 2

	// Calculate the number of slices in each sub-volume
	// This depends on the slice gap and the number of original slices
	slicesPerSubset := len(r.subSlices[0])
	slicesPerGap := int(r.params.SliceGap)
	if slicesPerGap < 1 {
		slicesPerGap = 1
	}

	quadDepth := (slicesPerSubset-1)*slicesPerGap + 1

	// Calculate full volume dimensions
	fullWidth := r.width
	fullHeight := r.height
	fullDepth := quadDepth * numSubsets / 2 // Divide by 2 because subsets overlap in z-direction

	// Create a 3D volume to hold the merged result
	mergedVolume := make([]float64, fullWidth*fullHeight*fullDepth)

	// Save the empty volume as intermediary result
	if r.params.SaveIntermediaryResults {
		// Save a few empty slices as examples
		sampleIndices := []int{0, fullDepth / 2, fullDepth - 1}
		for _, z := range sampleIndices {
			if z >= fullDepth {
				continue
			}

			slice := make([]float64, fullWidth*fullHeight)
			r.saveIntermediaryResult("05_merging/01_empty_volume", slice, z)
		}
	}

	// Merge the sub-volumes into the full volume
	fmt.Println("Merging sub-volumes...")

	// For each subset
	for s := 0; s < numSubsets; s++ {
		// Calculate z-offset for this subset
		zOffset := (s / 2) * quadDepth

		// For each quadrant in this subset
		for q := 0; q < numQuadrants; q++ {
			// Calculate x,y offsets for this quadrant
			xOffset := (q % 2) * quadWidth
			yOffset := (q / 2) * quadHeight

			// Get the sub-volume data
			subVolume := subVolumes[s][q]

			// Copy data from sub-volume to merged volume
			for z := 0; z < quadDepth; z++ {
				for y := 0; y < quadHeight; y++ {
					for x := 0; x < quadWidth; x++ {
						// Calculate source index in sub-volume
						srcIdx := z*quadWidth*quadHeight + y*quadWidth + x

						// Calculate destination index in merged volume
						dstX := xOffset + x
						dstY := yOffset + y
						dstZ := zOffset + z
						dstIdx := dstZ*fullWidth*fullHeight + dstY*fullWidth + dstX

						// Copy data if within bounds
						if srcIdx < len(subVolume) && dstIdx < len(mergedVolume) {
							mergedVolume[dstIdx] = subVolume[srcIdx]
						}
					}
				}
			}

			// Save intermediary result after merging each quadrant
			if r.params.SaveIntermediaryResults && q == numQuadrants-1 {
				// Save a few slices as examples after merging this subset
				sampleIndices := []int{zOffset, zOffset + quadDepth/2, zOffset + quadDepth - 1}
				for i, z := range sampleIndices {
					if z >= fullDepth {
						continue
					}

					slice := make([]float64, fullWidth*fullHeight)
					for y := 0; y < fullHeight; y++ {
						for x := 0; x < fullWidth; x++ {
							idx := z*fullWidth*fullHeight + y*fullWidth + x
							if idx < len(mergedVolume) {
								slice[y*fullWidth+x] = mergedVolume[idx]
							}
						}
					}

					stageName := fmt.Sprintf("05_merging/02_after_subset_%d", s)
					r.saveIntermediaryResult(stageName, slice, i)
				}
			}
		}
	}

	// Apply edge-preserving smoothing to the merged volume
	// This implements the mean-median logic from Algorithm 2 in the paper
	fmt.Println("Applying edge-preserving smoothing...")

	// Create a shearlet transform for edge detection during merging with configurable parameters
	scales := r.params.ShearletScales
	shears := r.params.ShearletShears
	coneParam := r.params.ShearletConeParam

	// Use default values if not specified
	if scales <= 0 {
		scales = 3
	}
	if shears <= 0 {
		shears = 8
	}
	if coneParam <= 0 {
		coneParam = 1.0
	}

	transform := shearlet.NewTransform(scales, shears, coneParam)

	// Process each slice in the merged volume
	for z := 0; z < fullDepth; z++ {
		// Extract the slice
		slice := make([]float64, fullWidth*fullHeight)
		for y := 0; y < fullHeight; y++ {
			for x := 0; x < fullWidth; x++ {
				idx := z*fullWidth*fullHeight + y*fullWidth + x
				if idx < len(mergedVolume) {
					slice[y*fullWidth+x] = mergedVolume[idx]
				}
			}
		}

		// Detect edges using shearlet transform
		edgeInfo := transform.DetectEdgesWithOrientation(slice)

		// Apply mean-median logic for edge preservation
		for y := 1; y < fullHeight-1; y++ {
			for x := 1; x < fullWidth-1; x++ {
				idx := y*fullWidth + x

				// Check if this is an edge pixel
				if edgeInfo.Edges[idx] > r.params.EdgeDetectionThreshold { // Use parameter for edge detection threshold
					// Get edge orientation
					orientation := edgeInfo.Orientations[idx]

					// Apply mean-median logic based on orientation
					// This is a simplified version of Algorithm 2 from the paper
					if orientation >= -math.Pi/4 && orientation < math.Pi/4 {
						// Horizontal edge - adjust pixels above and below
						above := []float64{
							slice[(y-1)*fullWidth+x-1],
							slice[(y-1)*fullWidth+x],
							slice[(y-1)*fullWidth+x+1],
						}
						below := []float64{
							slice[(y+1)*fullWidth+x-1],
							slice[(y+1)*fullWidth+x],
							slice[(y+1)*fullWidth+x+1],
						}

						// Apply median filter to preserve edges
						medianAbove := median(above)
						medianBelow := median(below)

						// Update the slice
						slice[(y-1)*fullWidth+x] = medianAbove
						slice[(y+1)*fullWidth+x] = medianBelow
					} else {
						// Vertical edge - adjust pixels to left and right
						left := []float64{
							slice[(y-1)*fullWidth+x-1],
							slice[y*fullWidth+x-1],
							slice[(y+1)*fullWidth+x-1],
						}
						right := []float64{
							slice[(y-1)*fullWidth+x+1],
							slice[y*fullWidth+x+1],
							slice[(y+1)*fullWidth+x+1],
						}

						// Apply median filter to preserve edges
						medianLeft := median(left)
						medianRight := median(right)

						// Update the slice
						slice[y*fullWidth+x-1] = medianLeft
						slice[y*fullWidth+x+1] = medianRight
					}
				}
			}
		}

		// Copy the processed slice back to the merged volume
		for y := 0; y < fullHeight; y++ {
			for x := 0; x < fullWidth; x++ {
				idx := z*fullWidth*fullHeight + y*fullWidth + x
				if idx < len(mergedVolume) {
					mergedVolume[idx] = slice[y*fullWidth+x]
				}
			}
		}

		// Save intermediary result for edge-preserved slices
		if r.params.SaveIntermediaryResults && (z == 0 || z == fullDepth/2 || z == fullDepth-1) {
			stageName := fmt.Sprintf("05_merging/03_edge_preserved")
			r.saveIntermediaryResult(stageName, slice, z)
		}
	}

	// Generate STL file using marching cubes
	fmt.Println("Generating STL file using marching cubes...")

	// Pre-process the volume to remove the floating plane
	// First, identify the main volume and any disconnected parts
	if r.params.Verbose {
		fmt.Println("Pre-processing volume to remove artifacts...")
	}

	// Find the min/max values to better determine an appropriate isoLevel
	minVal, maxVal := findMinMax(mergedVolume)
	if r.params.Verbose {
		fmt.Printf("Volume data range: min=%.3f, max=%.3f\n", minVal, maxVal)
	}

	// Calculate a better isoLevel based on the data range
	// Use 25% of the range from min to max as the threshold
	adaptiveIsoLevel := minVal + (maxVal-minVal)*r.params.IsoLevelPercent
	if r.params.Verbose {
		fmt.Printf("Using adaptive isoLevel: %.3f\n", adaptiveIsoLevel)
	}

	// Remove disconnected components (like the floating plane)
	// For each slice at the top and bottom, check for isolated regions
	// and connect them to the main volume or remove them

	// Process the top slices to remove the floating plane
	topSliceDepth := int(float64(fullDepth) * 0.15) // Process top 15% of slices
	for z := 0; z < topSliceDepth; z++ {
		for y := 0; y < fullHeight; y++ {
			for x := 0; x < fullWidth; x++ {
				idx := z*fullWidth*fullHeight + y*fullWidth + x
				if idx < len(mergedVolume) {
					// If this voxel is in the top 5% of the volume and has a high value
					// (likely part of the floating plane), reduce its value
					if z < int(float64(fullDepth)*0.05) && mergedVolume[idx] > adaptiveIsoLevel {
						// Significantly reduce the value to ensure it's below the isoLevel
						mergedVolume[idx] = minVal + (mergedVolume[idx]-minVal)*0.5
					}
				}
			}
		}
	}

	// Process the bottom slices to enhance the bottom part
	bottomSliceDepth := int(float64(fullDepth) * 0.15) // Process bottom 15% of slices
	for z := fullDepth - bottomSliceDepth; z < fullDepth; z++ {
		for y := 0; y < fullHeight; y++ {
			for x := 0; x < fullWidth; x++ {
				idx := z*fullWidth*fullHeight + y*fullWidth + x
				if idx < len(mergedVolume) {
					// Enhance values in the bottom part to ensure they're included
					if mergedVolume[idx] > minVal+(maxVal-minVal)*0.15 {
						// Boost values that are already somewhat significant
						mergedVolume[idx] = math.Min(mergedVolume[idx]*1.2, maxVal)
					}
				}
			}
		}
	}

	// Create marching cubes instance with the adaptive isoLevel
	mc := stl.NewMarchingCubes(mergedVolume, fullWidth, fullHeight, fullDepth, adaptiveIsoLevel)

	// Set scale based on slice gap
	// Ensure proper Z scaling to avoid flattening
	mc.SetScale(1.0, 1.0, float32(math.Max(r.params.SliceGap, 1.0)))

	// Generate triangles
	triangles := mc.GenerateTriangles()

	// Save STL file
	if err := stl.SaveToSTL(r.params.OutputFile, triangles); err != nil {
		return fmt.Errorf("failed to save STL file: %v", err)
	}

	return nil
}

// median calculates the median value of a slice of float64 values
func median(values []float64) float64 {
	// Create a copy to avoid modifying the original
	valuesCopy := make([]float64, len(values))
	copy(valuesCopy, values)

	// Sort the values
	sort.Float64s(valuesCopy)

	// Calculate median
	n := len(valuesCopy)
	if n == 0 {
		return 0
	}

	if n%2 == 0 {
		return (valuesCopy[n/2-1] + valuesCopy[n/2]) / 2
	}

	return valuesCopy[n/2]
}

// findMinMax returns the minimum and maximum values in a slice
func findMinMax(data []float64) (min, max float64) {
	if len(data) == 0 {
		return 0, 0
	}

	min = data[0]
	max = data[0]

	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	return min, max
}

// calculateEdgePreservation computes the edge preservation ratio
func (r *Reconstructor) calculateEdgePreservation(originalSlices []image.Image, subVolumes [][][]float64) float64 {
	// Handle empty inputs
	if len(originalSlices) == 0 || len(subVolumes) == 0 {
		return 1.0 // Default to 1 (perfect preservation) if no data to compare
	}

	// Create a shearlet transform for edge detection with configurable parameters
	scales := r.params.ShearletScales
	shears := r.params.ShearletShears
	coneParam := r.params.ShearletConeParam

	// Use default values if not specified
	if scales <= 0 {
		scales = 3
	}
	if shears <= 0 {
		shears = 8
	}
	if coneParam <= 0 {
		coneParam = 1.0
	}

	transformer := shearlet.NewTransform(scales, shears, coneParam)

	// Convert original slices to float array
	original := imagesToFloat(originalSlices)

	// Flatten reconstructed volumes for comparison
	var reconstructed []float64
	for subset := range subVolumes {
		for quadrant := range subVolumes[subset] {
			reconstructed = append(reconstructed, subVolumes[subset][quadrant]...)
		}
	}

	// Ensure we have data to compare
	if len(original) == 0 || len(reconstructed) == 0 {
		return 1.0
	}

	// Ensure matching lengths for comparison
	minLen := len(original)
	if len(reconstructed) < minLen {
		minLen = len(reconstructed)
	}

	// Ensure we have enough data for meaningful edge detection
	if minLen < 10 {
		return 1.0 // Not enough data for reliable edge detection
	}

	// Detect edges in both original and reconstructed data
	edgesOrig := transformer.DetectEdges(original[:minLen])
	edgesRecon := transformer.DetectEdges(reconstructed[:minLen])

	// Check if we have valid edge data
	if len(edgesOrig) == 0 || len(edgesRecon) == 0 {
		return 1.0
	}

	// Verify variance in edge data - if no edges are detected, correlation will be NaN
	var hasVarianceOrig, hasVarianceRecon bool
	for i := 1; i < len(edgesOrig); i++ {
		if edgesOrig[i] != edgesOrig[0] {
			hasVarianceOrig = true
			break
		}
	}

	for i := 1; i < len(edgesRecon); i++ {
		if edgesRecon[i] != edgesRecon[0] {
			hasVarianceRecon = true
			break
		}
	}

	// If either dataset has no variance, correlation is undefined
	if !hasVarianceOrig || !hasVarianceRecon {
		return 1.0 // Default to 1.0 when correlation is undefined
	}

	// Calculate correlation between edge maps using Gonum
	correlation := stat.Correlation(edgesOrig, edgesRecon, nil)

	// Handle potential NaN or infinity results
	if math.IsNaN(correlation) || math.IsInf(correlation, 0) {
		return 1.0
	}

	// Ensure the result is in the valid range [0,1]
	// Since correlation can be negative, we take absolute value and cap at 1
	return math.Min(math.Abs(correlation), 1.0)
}

// GetVolumeData returns the reconstructed volume data and its dimensions
// This method is used to access the volume data for visualization
func (r *Reconstructor) GetVolumeData() ([]float64, int, int, int) {
	// Calculate dimensions of the full volume
	sliceWidth := r.width
	sliceHeight := r.height
	numSlices := len(r.slices)

	// Calculate the number of interpolated slices based on slice gap
	slicesPerGap := int(math.Ceil(r.params.SliceGap))
	totalDepth := (numSlices-1)*slicesPerGap + 1

	// Create a full volume array
	volumeData := make([]float64, sliceWidth*sliceHeight*totalDepth)

	// Process the volume in parallel to reconstruct it
	var wg sync.WaitGroup
	numCores := r.params.NumCores

	// Divide the work among available cores
	slicesPerCore := (numSlices + numCores - 1) / numCores

	for c := 0; c < numCores; c++ {
		wg.Add(1)

		go func(coreID int) {
			defer wg.Done()

			// Calculate slice range for this core
			startSlice := coreID * slicesPerCore
			endSlice := (coreID + 1) * slicesPerCore
			if endSlice > numSlices {
				endSlice = numSlices
			}

			// Skip if no slices to process
			if startSlice >= numSlices {
				return
			}

			// Process each slice
			for i := startSlice; i < endSlice; i++ {
				// Convert slice to float data
				sliceData := imageToFloat(r.slices[i])

				// Calculate position in volume
				zPos := i * slicesPerGap

				// Copy slice data to volume
				for y := 0; y < sliceHeight; y++ {
					for x := 0; x < sliceWidth; x++ {
						srcIdx := y*sliceWidth + x
						dstIdx := zPos*sliceWidth*sliceHeight + y*sliceWidth + x

						if dstIdx < len(volumeData) && srcIdx < len(sliceData) {
							volumeData[dstIdx] = sliceData[srcIdx]
						}
					}
				}

				// Interpolate between slices if not the last slice
				if i < numSlices-1 {
					nextSliceData := imageToFloat(r.slices[i+1])

					// Interpolate for each gap position
					for z := 1; z < slicesPerGap; z++ {
						t := float64(z) / float64(slicesPerGap)

						for y := 0; y < sliceHeight; y++ {
							for x := 0; x < sliceWidth; x++ {
								srcIdx := y*sliceWidth + x
								dstIdx := (zPos+z)*sliceWidth*sliceHeight + y*sliceWidth + x

								if dstIdx < len(volumeData) && srcIdx < len(sliceData) && srcIdx < len(nextSliceData) {
									// Linear interpolation
									volumeData[dstIdx] = (1-t)*sliceData[srcIdx] + t*nextSliceData[srcIdx]
								}
							}
						}
					}
				}
			}
		}(c)
	}

	// Wait for all cores to finish
	wg.Wait()

	return volumeData, sliceWidth, sliceHeight, totalDepth
}

// loadImage loads an image from a file
func loadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, err := jpeg.Decode(file)
	if err != nil {
		return nil, err
	}

	return img, nil
}

// imageToFloat converts a single image to float array
func imageToFloat(img image.Image) []float64 {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	result := make([]float64, width*height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, _, _, _ := img.At(x, y).RGBA()
			// Convert 16-bit color to float64 (0-1 range)
			result[y*width+x] = float64(r) / 65535.0
		}
	}

	return result
}

// imagesToFloat converts a slice of images to float array
func imagesToFloat(images []image.Image) []float64 {
	if len(images) == 0 {
		return nil
	}

	bounds := images[0].Bounds()
	width := bounds.Dx()
	height := bounds.Dy()
	size := width * height
	result := make([]float64, size*len(images))

	for i, img := range images {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				r, _, _, _ := img.At(x, y).RGBA()
				// Convert 16-bit color to float64 (0-1 range)
				result[i*size+y*width+x] = float64(r) / 65535.0
			}
		}
	}

	return result
}

// floatToImage converts a float array back to an image
func floatToImage(data []float64, width, height int) image.Image {
	img := image.NewGray16(image.Rect(0, 0, width, height))

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*width + x
			if idx < len(data) {
				// Convert 0-1 range to 16-bit grayscale
				value := uint16(data[idx] * 65535.0)
				img.Set(x, y, color.Gray16{Y: value})
			}
		}
	}

	return img
}

// saveIntermediaryResult saves an intermediary result during the reconstruction process.
// This helps visualize the steps of the algorithm and debug the reconstruction process.
func (r *Reconstructor) saveIntermediaryResult(stage string, data interface{}, index int) error {
	// Skip if saving intermediary results is disabled
	if !r.params.SaveIntermediaryResults {
		return nil
	}

	// Create the intermediary directory if it doesn't exist
	stageDir := filepath.Join(r.params.IntermediaryDir, stage)
	if err := os.MkdirAll(stageDir, 0755); err != nil {
		return fmt.Errorf("failed to create intermediary directory: %v", err)
	}

	// Handle different types of data
	switch v := data.(type) {
	case image.Image:
		// Save as JPEG image
		filename := filepath.Join(stageDir, fmt.Sprintf("%03d.jpg", index))
		file, err := os.Create(filename)
		if err != nil {
			return fmt.Errorf("failed to create image file: %v", err)
		}
		defer file.Close()

		if err := jpeg.Encode(file, v, &jpeg.Options{Quality: 90}); err != nil {
			return fmt.Errorf("failed to encode image: %v", err)
		}

	case []float64:
		// For float slices, convert to image if it's 2D data
		if len(v) == r.width*r.height {
			// It's a 2D slice, convert to image
			img := floatToImage(v, r.width, r.height)
			filename := filepath.Join(stageDir, fmt.Sprintf("%03d.jpg", index))
			file, err := os.Create(filename)
			if err != nil {
				return fmt.Errorf("failed to create image file: %v", err)
			}
			defer file.Close()

			if err := jpeg.Encode(file, img, &jpeg.Options{Quality: 90}); err != nil {
				return fmt.Errorf("failed to encode image: %v", err)
			}
		} else {
			// For 3D data or other float slices, save as binary file
			filename := filepath.Join(stageDir, fmt.Sprintf("%03d.bin", index))
			file, err := os.Create(filename)
			if err != nil {
				return fmt.Errorf("failed to create binary file: %v", err)
			}
			defer file.Close()

			// Write data as binary
			for _, val := range v {
				if err := binary.Write(file, binary.LittleEndian, val); err != nil {
					return fmt.Errorf("failed to write binary data: %v", err)
				}
			}
		}

	case [][][]float64:
		// For 3D volume data, save each slice as an image
		for z := 0; z < len(v); z++ {
			for y := 0; y < len(v[z]); y++ {
				for x := 0; x < len(v[z][y]); x++ {
					// Create a flat slice for this z-level
					slice := make([]float64, r.width*r.height)
					for y := 0; y < r.height; y++ {
						for x := 0; x < r.width; x++ {
							if y < len(v[z]) && x < len(v[z][y]) {
								slice[y*r.width+x] = v[z][y][x]
							}
						}
					}

					// Save this slice
					sliceDir := filepath.Join(stageDir, fmt.Sprintf("slice_%03d", z))
					if err := os.MkdirAll(sliceDir, 0755); err != nil {
						return fmt.Errorf("failed to create slice directory: %v", err)
					}

					img := floatToImage(slice, r.width, r.height)
					filename := filepath.Join(sliceDir, fmt.Sprintf("%03d.jpg", index))
					file, err := os.Create(filename)
					if err != nil {
						return fmt.Errorf("failed to create image file: %v", err)
					}

					if err := jpeg.Encode(file, img, &jpeg.Options{Quality: 90}); err != nil {
						file.Close()
						return fmt.Errorf("failed to encode image: %v", err)
					}
					file.Close()
				}
			}
		}

	default:
		// For other types, save as text representation
		filename := filepath.Join(stageDir, fmt.Sprintf("%03d.txt", index))
		file, err := os.Create(filename)
		if err != nil {
			return fmt.Errorf("failed to create text file: %v", err)
		}
		defer file.Close()

		fmt.Fprintf(file, "%v", v)
	}

	return nil
}
