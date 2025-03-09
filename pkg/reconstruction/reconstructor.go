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

	"mrislicesto3d/pkg/interpolation"
	"mrislicesto3d/pkg/shearlet"
	"mrislicesto3d/pkg/stl"
)

// ValidationMetrics holds the reconstruction quality metrics used in the paper.
// These metrics are used to evaluate the quality of the reconstruction against
// ground truth data or to compare different reconstruction methods.
type ValidationMetrics struct {
	// MI (Mutual Information) measures the statistical dependency between
	// the original and reconstructed data. Higher values indicate better
	// information preservation.
	MI float64
	
	// EntropyDiff is the difference in information content (entropy) between
	// the original and reconstructed volumes. Lower values indicate better
	// information preservation.
	EntropyDiff float64
	
	// RMSE (Root Mean Square Error) measures the average squared difference
	// between original and reconstructed voxel intensities. Lower values
	// indicate better reconstruction fidelity.
	RMSE float64
	
	// SSIM (Structural Similarity Index) measures the perceived similarity
	// between original and reconstructed images, considering luminance,
	// contrast, and structure. Values range from -1 to 1, with 1 indicating
	// perfect similarity.
	SSIM float64
	
	// EdgePreserved measures how well edges and boundaries are maintained
	// in the reconstruction. Values range from 0 to 1, with 1 indicating
	// perfect edge preservation.
	EdgePreserved float64
	
	// Accuracy is the overall combined metric as defined in the paper,
	// calculated from the other metrics to provide a single quality score.
	// Values range from 0 to 100%, with higher values indicating better
	// overall reconstruction quality.
	Accuracy float64
}

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
	
	// metrics stores the quality assessment metrics after reconstruction
	metrics ValidationMetrics
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
	// Create intermediary directory if needed
	if r.params.SaveIntermediaryResults {
		if err := os.MkdirAll(r.params.IntermediaryDir, 0755); err != nil {
			return fmt.Errorf("failed to create intermediary directory: %v", err)
		}
	}

	// Step 1: Load and preprocess input slices
	fmt.Println("Step 1: Loading input slices...")
	if err := r.loadSlices(); err != nil {
		return fmt.Errorf("failed to load slices: %v", err)
	}
	
	// Save original slices
	if r.params.SaveIntermediaryResults {
		fmt.Println("Saving original slices...")
		for i, slice := range r.slices {
			if err := r.saveIntermediaryResult("01_original_slices", slice, i); err != nil {
				fmt.Printf("Warning: Failed to save original slice %d: %v\n", i, err)
			}
		}
	}

	// Step 2: Apply shearlet transform for denoising
	fmt.Println("Step 2: Applying shearlet transform for denoising...")
	if err := r.denoiseSlices(); err != nil {
		return fmt.Errorf("failed to denoise slices: %v", err)
	}
	
	// Save denoised slices
	if r.params.SaveIntermediaryResults {
		fmt.Println("Saving denoised slices...")
		for i, slice := range r.slices {
			if err := r.saveIntermediaryResult("02_denoised_slices", slice, i); err != nil {
				fmt.Printf("Warning: Failed to save denoised slice %d: %v\n", i, err)
			}
		}
	}

	// Step 3: Divide dataset into quadrants for parallel processing
	fmt.Println("Step 3: Dividing dataset for parallel processing...")
	if err := r.divideDataset(); err != nil {
		return fmt.Errorf("failed to divide dataset: %v", err)
	}
	
	// Save divided dataset
	if r.params.SaveIntermediaryResults {
		fmt.Println("Saving divided dataset...")
		for i, subset := range r.subSlices {
			for j, slice := range subset {
				for k, quadrant := range slice {
					stageName := fmt.Sprintf("03_divided_dataset/subset_%d/slice_%d", i, j)
					if err := r.saveIntermediaryResult(stageName, quadrant, k); err != nil {
						fmt.Printf("Warning: Failed to save quadrant %d of slice %d in subset %d: %v\n", 
							k, j, i, err)
					}
				}
			}
		}
	}

	// Step 4: Process sub-volumes in parallel
	fmt.Println("Step 4: Processing sub-volumes with edge-preserved kriging...")
	subVolumes, err := r.processSubVolumesInParallel()
	if err != nil {
		return fmt.Errorf("failed to process sub-volumes: %v", err)
	}
	
	// Save sub-volumes
	if r.params.SaveIntermediaryResults {
		fmt.Println("Saving processed sub-volumes...")
		for i, subVolume := range subVolumes {
			stageName := fmt.Sprintf("04_processed_subvolumes")
			if err := r.saveIntermediaryResult(stageName, subVolume, i); err != nil {
				fmt.Printf("Warning: Failed to save sub-volume %d: %v\n", i, err)
			}
		}
	}

	// Step 5: Merge sub-volumes and generate 3D mesh
	fmt.Println("Step 5: Merging sub-volumes and generating 3D mesh...")
	if err := r.mergeAndGenerateSTL(subVolumes); err != nil {
		return fmt.Errorf("failed to merge and generate STL: %v", err)
	}
	
	// Save merged volume
	if r.params.SaveIntermediaryResults {
		fmt.Println("Saving merged volume slices...")
		volumeData, width, height, depth := r.GetVolumeData()
		
		// Save slices of the merged volume
		for z := 0; z < depth; z += 1 {
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
				fmt.Printf("Warning: Failed to save merged volume slice %d: %v\n", z, err)
			}
		}
	}

	// Step 6: Calculate validation metrics
	fmt.Println("Step 6: Calculating validation metrics...")
	r.calculateValidationMetrics(subVolumes)

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
	// Read input directory
	files, err := ioutil.ReadDir(r.params.InputDir)
	if err != nil {
		return err
	}

	// Filter and sort JPG files
	var imageFiles []string
	for _, file := range files {
		ext := strings.ToLower(filepath.Ext(file.Name()))
		if ext == ".jpg" || ext == ".jpeg" {
			imageFiles = append(imageFiles, file.Name())
		}
	}

	if len(imageFiles) == 0 {
		return fmt.Errorf("no JPG images found in input directory")
	}

	// Sort files alphanumerically to ensure correct slice order
	// This is important for the sequential nature of MRI slices,
	// as it maintains the proper anatomical ordering of structures
	sort.Slice(imageFiles, func(i, j int) bool {
		// Extract numbers from filenames to determine the slice order
		numI := extractNumber(imageFiles[i])
		numJ := extractNumber(imageFiles[j])
		return numI < numJ
	})

	// Load each image and add it to the slice collection
	for _, filename := range imageFiles {
		img, err := loadImage(filepath.Join(r.params.InputDir, filename))
		if err != nil {
			return fmt.Errorf("failed to load image %s: %v", filename, err)
		}

		// Store dimensions from first image
		// We assume all slices have the same dimensions
		if len(r.slices) == 0 {
			bounds := img.Bounds()
			r.width = bounds.Dx()
			r.height = bounds.Dy()
		}

		r.slices = append(r.slices, img)
	}

	// Log information about the loaded dataset
	fmt.Printf("Loaded %d slices with dimensions %dx%d\n", len(r.slices), r.width, r.height)
	fmt.Printf("Inter-slice gap: %.1f mm\n", r.params.SliceGap)
	
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
// 1. For each MRI slice:
//    a. Convert the image to a float array for mathematical processing
//    b. Apply the edge-preserved smoothing algorithm using Shearlet coefficients
//    c. Convert the processed data back to an image format
//
// Unlike traditional denoising methods that might blur important anatomical boundaries,
// this approach preserves edge information that is critical for accurate 3D reconstruction.
//
// Returns:
//   - nil if successful (processing is done in-place on the slice collection)
func (r *Reconstructor) denoiseSlices() error {
	fmt.Println("Applying Shearlet transform for denoising...")
	
	// Create a new Shearlet transform instance
	// This will be used for all slices to maintain consistency
	transformer := shearlet.NewTransform()
	
	// Process each slice sequentially
	for i, img := range r.slices {
		// Convert image to float array for mathematical processing
		// (Shearlet transform operates on continuous values)
		imgFloat := imageToFloat(img)
		
		// Apply edge-preserving denoising using Shearlet coefficients
		// This preserves important structural boundaries while removing noise
		denoised := transformer.ApplyEdgePreservedSmoothing(imgFloat)
		
		// Convert the processed data back to image format
		// and replace the original slice with the denoised version
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
	numSubsets := len(r.subSlices)
	numQuadrants := 4 // Always 4 quadrants as per paper
	
	// Create result array for processed sub-volumes
	result := make([][][]float64, numSubsets)
	for i := range result {
		result[i] = make([][]float64, numQuadrants)
	}
	
	// Create a channel for results
	type processingResult struct {
		subsetIdx   int
		quadrantIdx int
		data        []float64
		err         error
	}
	resultChan := make(chan processingResult)
	
	// Count total tasks for progress tracking
	totalTasks := numSubsets * numQuadrants
	completedTasks := 0
	
	// Process each sub-volume in parallel
	for i := 0; i < numSubsets; i++ {
		for j := 0; j < numQuadrants; j++ {
			// Extract slices for this quadrant
			slices := make([]image.Image, len(r.subSlices[i]))
			for k := range r.subSlices[i] {
				slices[k] = r.subSlices[i][k][j]
			}
			
			// Process in a goroutine
			go func(subsetIdx, quadrantIdx int, quadrantSlices []image.Image) {
				// Process this sub-volume
				data, err := r.processSubVolume(quadrantSlices)
				
				// Save intermediary results if enabled
				if r.params.SaveIntermediaryResults {
					stageName := fmt.Sprintf("04_edge_preserved_kriging/subset_%d/quadrant_%d", subsetIdx, quadrantIdx)
					
					// Save the first few slices as examples
					sliceWidth := r.width / 2  // Half width for quadrants
					sliceHeight := r.height / 2 // Half height for quadrants
					
					// Calculate the number of slices in the interpolated volume
					slicesPerGap := int(r.params.SliceGap)
					if slicesPerGap < 1 {
						slicesPerGap = 1
					}
					totalSlices := (len(quadrantSlices) - 1) * slicesPerGap + 1
					
					// Save a sample of slices (first, middle, last)
					sampleIndices := []int{0, totalSlices / 2, totalSlices - 1}
					for _, z := range sampleIndices {
						if z >= totalSlices {
							continue
						}
						
						// Extract slice from the 3D volume
						slice := make([]float64, sliceWidth*sliceHeight)
						for y := 0; y < sliceHeight; y++ {
							for x := 0; x < sliceWidth; x++ {
								idx := z*sliceWidth*sliceHeight + y*sliceWidth + x
								if idx < len(data) {
									slice[y*sliceWidth+x] = data[idx]
								}
							}
						}
						
						// Save this slice
						r.saveIntermediaryResult(stageName, slice, z)
					}
				}
				
				// Send result back through channel
				resultChan <- processingResult{
					subsetIdx:   subsetIdx,
					quadrantIdx: quadrantIdx,
					data:        data,
					err:         err,
				}
			}(i, j, slices)
		}
	}
	
	// Collect results
	for completedTasks < totalTasks {
		res := <-resultChan
		completedTasks++
		
		// Check for errors
		if res.err != nil {
			return nil, fmt.Errorf("sub-volume processing failed: %v", res.err)
		}
		
		// Store result
		result[res.subsetIdx][res.quadrantIdx] = res.data
		
		// Print progress
		progress := float64(completedTasks) / float64(totalTasks) * 100
		fmt.Printf("\rProcessing sub-volumes: %.1f%% complete", progress)
	}
	fmt.Println() // New line after progress
	
	return result, nil
}

// processSubVolume processes a single sub-volume using edge-preserved kriging
// as described in Algorithm 2 of the paper
func (r *Reconstructor) processSubVolume(slices []image.Image) ([]float64, error) {
	// Convert images to float arrays
	data := imagesToFloat(slices)
	
	// Step 1: Apply edge-preserved kriging interpolation
	// This follows Algorithm 2 from the paper
	fmt.Println("Applying edge-preserved kriging interpolation...")
	interpolator := interpolation.NewKriging(data, r.params.SliceGap)
	interpolated, err := interpolator.Interpolate()
	if err != nil {
		return nil, fmt.Errorf("kriging interpolation failed: %v", err)
	}
	
	// Step 2: Apply Shearlet transform for edge detection on the YZ plane
	// as described in section 4.1.2 of the paper
	fmt.Println("Applying Shearlet transform for edge detection...")
	transformer := shearlet.NewTransform()
	
	// Skip edge detection for now - we'll use it directly in ApplyEdgePreservedSmoothing
	// which already calls DetectEdgesWithOrientation internally
	
	// Step 3: Apply edge-preserved smoothing with mean-median logic
	// This implements the "Edge preserved Kriging interpolation" algorithm
	fmt.Println("Applying edge-preserved smoothing...")
	smoothed := transformer.ApplyEdgePreservedSmoothing(interpolated)
	
	return smoothed, nil
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
	
	quadDepth := (slicesPerSubset - 1) * slicesPerGap + 1
	
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
	
	// Create a shearlet transform instance for edge detection
	transform := shearlet.NewTransform()
	
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
				if edgeInfo.Edges[idx] > 0.5 { // Threshold for edge detection
					// Get edge orientation
					orientation := edgeInfo.Orientations[idx]
					
					// Apply mean-median logic based on orientation
					// This is a simplified version of Algorithm 2 from the paper
					if orientation >= -math.Pi/4 && orientation < math.Pi/4 {
						// Horizontal edge - adjust pixels above and below
						above := []float64{
							slice[(y-1)*fullWidth + x-1],
							slice[(y-1)*fullWidth + x],
							slice[(y-1)*fullWidth + x+1],
						}
						below := []float64{
							slice[(y+1)*fullWidth + x-1],
							slice[(y+1)*fullWidth + x],
							slice[(y+1)*fullWidth + x+1],
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
							slice[(y-1)*fullWidth + x-1],
							slice[y*fullWidth + x-1],
							slice[(y+1)*fullWidth + x-1],
						}
						right := []float64{
							slice[(y-1)*fullWidth + x+1],
							slice[y*fullWidth + x+1],
							slice[(y+1)*fullWidth + x+1],
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
	
	// Create marching cubes instance
	mc := stl.NewMarchingCubes(mergedVolume, fullWidth, fullHeight, fullDepth, 0.5)
	
	// Set scale based on slice gap
	mc.SetScale(1.0, 1.0, float32(r.params.SliceGap))
	
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

// calculateValidationMetrics computes quality metrics as described in the paper
func (r *Reconstructor) calculateValidationMetrics(subVolumes [][][]float64) {
	fmt.Println("Calculating validation metrics...")
	
	// Initialize metrics
	var totalMI, totalRMSE, totalSSIM, totalEntropy float64
	count := 0
	
	// Process each quadrant in each subset
	for subset := range subVolumes {
		for quadrant := range subVolumes[subset] {
			// Get reconstructed data
			reconstructed := subVolumes[subset][quadrant]
			
			// Get original slice data for this subset/quadrant
			var originalSlices []image.Image
			for _, slices := range r.subSlices[subset] {
				originalSlices = append(originalSlices, slices[quadrant])
			}
			original := imagesToFloat(originalSlices)
			
			// Calculate metrics
			mi := calculateMutualInformation(original, reconstructed)
			rmse := calculateRMSE(original, reconstructed)
			ssim := calculateSSIM(original, reconstructed)
			entropy := calculateEntropyDifference(original, reconstructed)
			
			// Accumulate metrics
			totalMI += mi
			totalRMSE += rmse
			totalSSIM += ssim
			totalEntropy += entropy
			count++
		}
	}
	
	// Calculate average metrics
	if count > 0 {
		r.metrics.MI = totalMI / float64(count)
		r.metrics.RMSE = totalRMSE / float64(count)
		r.metrics.SSIM = totalSSIM / float64(count)
		r.metrics.EntropyDiff = totalEntropy / float64(count)
		
		// Calculate edge preservation ratio
		r.metrics.EdgePreserved = calculateEdgePreservation(r.slices, subVolumes)
		
		// Calculate overall accuracy as defined in equation (2) of the paper
		r.metrics.Accuracy = (1 - r.metrics.EntropyDiff) * 
		                     (1 - (1 - r.metrics.MI)) * 
		                     (1 - r.metrics.RMSE) * 
		                     r.metrics.SSIM * 
		                     r.metrics.EdgePreserved
		r.metrics.Accuracy *= 100 // Convert to percentage
	}
}

// calculateMutualInformation computes the mutual information between two datasets
func calculateMutualInformation(original, reconstructed []float64) float64 {
	// Simple implementation - in real code would use binning and histogram approach
	n := len(original)
	if n != len(reconstructed) || n == 0 {
		return 0
	}
	
	// Calculate means
	meanOrig := 0.0
	meanRecon := 0.0
	for i := 0; i < n; i++ {
		meanOrig += original[i]
		meanRecon += reconstructed[i]
	}
	meanOrig /= float64(n)
	meanRecon /= float64(n)
	
	// Calculate covariance and variances
	covar := 0.0
	varOrig := 0.0
	varRecon := 0.0
	for i := 0; i < n; i++ {
		diffOrig := original[i] - meanOrig
		diffRecon := reconstructed[i] - meanRecon
		covar += diffOrig * diffRecon
		varOrig += diffOrig * diffOrig
		varRecon += diffRecon * diffRecon
	}
	
	// Normalize
	covar /= float64(n)
	varOrig /= float64(n)
	varRecon /= float64(n)
	
	// Calculate mutual information approximation
	// MI ≈ 0.5 * log(var(X) * var(Y) / (var(X) * var(Y) - cov(X,Y)²))
	if varOrig > 0 && varRecon > 0 {
		determinant := varOrig * varRecon - covar*covar
		if determinant > 0 {
			return 0.5 * math.Log(varOrig * varRecon / determinant)
		}
	}
	
	return 0
}

// calculateRMSE computes the root mean square error
func calculateRMSE(original, reconstructed []float64) float64 {
	n := len(original)
	if n != len(reconstructed) || n == 0 {
		return 0
	}
	
	// Calculate MSE
	mse := 0.0
	for i := 0; i < n; i++ {
		diff := original[i] - reconstructed[i]
		mse += diff * diff
	}
	mse /= float64(n)
	
	// Return RMSE
	return math.Sqrt(mse)
}

// calculateSSIM computes the Structural Similarity Index
func calculateSSIM(original, reconstructed []float64) float64 {
	// Constants for SSIM calculation
	const L = 1.0 // Dynamic range
	const k1 = 0.01
	const k2 = 0.03
	
	c1 := (k1 * L) * (k1 * L)
	c2 := (k2 * L) * (k2 * L)
	
	n := len(original)
	if n != len(reconstructed) || n == 0 {
		return 0
	}
	
	// Calculate means using Gonum
	muX := stat.Mean(original, nil)
	muY := stat.Mean(reconstructed, nil)
	
	// Calculate variances and covariance using Gonum
	sigmaX := stat.Variance(original, nil)
	sigmaY := stat.Variance(reconstructed, nil)
	sigmaXY := stat.Covariance(original, reconstructed, nil)
	
	// Calculate SSIM
	num := (2*muX*muY + c1) * (2*sigmaXY + c2)
	den := (muX*muX + muY*muY + c1) * (sigmaX + sigmaY + c2)
	
	if den > 0 {
		return num / den
	}
	return 0
}

// calculateEntropyDifference computes the entropy difference
func calculateEntropyDifference(original, reconstructed []float64) float64 {
	n := len(original)
	if n != len(reconstructed) || n == 0 {
		return 0
	}
	
	// Calculate entropy for original and reconstructed data
	entropyOrig := calculateEntropy(original)
	entropyRecon := calculateEntropy(reconstructed)
	
	// Return absolute difference
	return math.Abs(entropyOrig - entropyRecon)
}

// calculateEntropy computes the Shannon entropy of data
func calculateEntropy(data []float64) float64 {
	n := len(data)
	if n == 0 {
		return 0
	}
	
	// Find min/max values
	min, max := findMinMax(data)
	
	// If all values are the same, entropy is 0
	if max <= min {
		return 0
	}
	
	// Create histogram with 256 bins
	const numBins = 256
	
	// Create histogram manually to avoid edge cases with Gonum's Histogram
	hist := make([]float64, numBins)
	binWidth := (max - min) / float64(numBins)
	
	for _, v := range data {
		// Calculate bin index, ensuring it's within bounds
		binIdx := int((v - min) / binWidth)
		if binIdx >= numBins {
			binIdx = numBins - 1
		} else if binIdx < 0 {
			binIdx = 0
		}
		hist[binIdx]++
	}
	
	// Calculate entropy
	entropy := 0.0
	for _, count := range hist {
		if count > 0 {
			p := count / float64(n)
			entropy -= p * math.Log2(p)
		}
	}
	
	return entropy
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
func calculateEdgePreservation(originalSlices []image.Image, subVolumes [][][]float64) float64 {
	transformer := shearlet.NewTransform()
	original := imagesToFloat(originalSlices)
	
	// Flatten reconstructed volumes for comparison
	var reconstructed []float64
	for subset := range subVolumes {
		for quadrant := range subVolumes[subset] {
			reconstructed = append(reconstructed, subVolumes[subset][quadrant]...)
		}
	}
	
	// Ensure matching lengths for comparison
	minLen := len(original)
	if len(reconstructed) < minLen {
		minLen = len(reconstructed)
	}
	
	// Detect edges in both original and reconstructed data
	edgesOrig := transformer.DetectEdges(original[:minLen])
	edgesRecon := transformer.DetectEdges(reconstructed[:minLen])
	
	// Calculate correlation between edge maps using Gonum
	correlation := stat.Correlation(edgesOrig, edgesRecon, nil)
	
	return correlation
}

// GetMetrics returns the current validation metrics
func (r *Reconstructor) GetMetrics() ValidationMetrics {
	return r.metrics
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
	totalDepth := (numSlices - 1) * slicesPerGap + 1
	
	// Create a full volume array
	volumeData := make([]float64, sliceWidth * sliceHeight * totalDepth)
	
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

// Helper functions

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