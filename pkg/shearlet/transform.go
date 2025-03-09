package shearlet

import (
	"fmt"
	"math"
	"math/cmplx"
)

// Transform implements the discrete shearlet transform as described in the paper
// "Fast 3D Volumetric Image Reconstruction from 2D MRI Slices by Parallel Processing".
//
// The shearlet transform is a multiscale and multidirectional transform that
// is particularly effective at representing images with anisotropic features
// such as edges and contours in medical images. It extends the wavelet transform
// by adding directional sensitivity, making it ideal for edge detection and
// edge-preserving denoising in MRI data.
//
// This implementation includes:
// - Multiscale decomposition (via the scales parameter)
// - Directional sensitivity (via the shears parameter)
// - Frequency domain processing for efficient computation
// - Edge detection and orientation estimation
// - Edge-preserved smoothing for noise reduction
type Transform struct {
	// scales defines the number of scale levels in the transform
	// Each scale represents a different level of detail, from coarse to fine
	scales int
	
	// shears defines the base number of directional shears per scale
	// Higher scales use more shears to provide finer directional resolution
	shears int
	
	// mother holds the mother wavelet coefficients (not currently used)
	mother []complex128
	
	// psi contains the shearlet generators for each scale
	// These are the frequency domain filters that provide directional sensitivity
	psi [][]complex128
	
	// phi is the scaling function (low-pass filter) used to capture
	// the coarse-scale approximation of the image
	phi []complex128
	
	// coneParam controls the partitioning of the frequency domain into
	// horizontal and vertical cone regions for directional processing
	coneParam float64
}

// EdgeInfo holds edge detection information
type EdgeInfo struct {
	Edges       []float64  // Edge strength map
	Orientations []float64 // Edge orientations in radians
}

// NewTransform creates a new shearlet transform instance
func NewTransform() *Transform {
	t := &Transform{
		scales:     4,   // Number of scales
		shears:     8,   // Number of shear parameters per scale
		coneParam:  1.0, // Default cone parameter
	}
	t.initializeGenerators()
	return t
}

// initializeGenerators creates shearlet generators for each scale
func (t *Transform) initializeGenerators() {
	size := 32 // Base size for the generators
	t.psi = make([][]complex128, t.scales)
	
	// Initialize scaling function (low-pass filter)
	t.phi = t.createScalingFunction(size)
	
	// Create shearlet generators for each scale
	for j := 0; j < t.scales; j++ {
		t.psi[j] = t.createShearletGenerator(size, j)
	}
}

// createScalingFunction generates the scaling function (father wavelet)
func (t *Transform) createScalingFunction(size int) []complex128 {
	phi := make([]complex128, size*size)
	
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			// Map to frequency domain coordinates
			ω1 := 2 * math.Pi * float64(i-size/2) / float64(size)
			ω2 := 2 * math.Pi * float64(j-size/2) / float64(size)
			r := math.Sqrt(ω1*ω1 + ω2*ω2)
			
			// Meyer-like scaling function
			var value float64
			if r <= math.Pi/4 {
				value = 1
			} else if r <= math.Pi/2 {
				t := (4/math.Pi)*r - 1
				value = math.Cos(math.Pi/2 * meyer(t))
			}
			
			phi[i*size+j] = complex(value, 0)
		}
	}
	
	return phi
}

// createShearletGenerator creates a shearlet generator for a specific scale
func (t *Transform) createShearletGenerator(size int, scale int) []complex128 {
	psi := make([]complex128, size*size)
	
	// Scale-dependent parameters
	a := math.Pow(2, float64(-2*scale)) // Parabolic scaling
	
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			// Map to frequency domain coordinates
			ω1 := 2 * math.Pi * float64(i-size/2) / float64(size)
			ω2 := 2 * math.Pi * float64(j-size/2) / float64(size)
			
			// Apply cone-adapted coordinates
			r := math.Sqrt(ω1*ω1 + ω2*ω2)
			θ := math.Atan2(ω2, ω1)
			
			// Radial component (bandpass)
			var radial float64
			if r >= math.Pi/4 && r <= math.Pi {
				t := (4/math.Pi)*r - 1
				radial = meyer(t)
			}
			
			// Angular component (directional)
			angular := math.Exp(-θ*θ/(2*a)) // Gaussian directional window
			
			// Combine components
			value := radial * angular
			
			psi[i*size+j] = complex(value, 0)
		}
	}
	
	return psi
}

// DetectEdgesWithOrientation applies the shearlet transform to detect edges and their orientations
// as described in the paper
func (t *Transform) DetectEdgesWithOrientation(data []float64) EdgeInfo {
	n := len(data)
	size := int(math.Sqrt(float64(n)))
	if size*size != n {
		return EdgeInfo{
			Edges:       make([]float64, n),
			Orientations: make([]float64, n),
		}
	}

	fmt.Printf("   Detecting edges in %dx%d image...\n", size, size)

	// For synthetic images with sharp edges, we can enhance the detection
	// by adding a pre-processing step to detect gradient magnitude
	gradientEdges := make([]float64, n)
	for y := 1; y < size-1; y++ {
		for x := 1; x < size-1; x++ {
			// Simple Sobel-like gradient calculation
			gx := data[(y-1)*size+(x+1)] + 2*data[y*size+(x+1)] + data[(y+1)*size+(x+1)] -
				 data[(y-1)*size+(x-1)] - 2*data[y*size+(x-1)] - data[(y+1)*size+(x-1)]
			
			gy := data[(y-1)*size+(x-1)] + 2*data[(y-1)*size+x] + data[(y-1)*size+(x+1)] -
				 data[(y+1)*size+(x-1)] - 2*data[(y+1)*size+x] - data[(y+1)*size+(x+1)]
			
			// Gradient magnitude
			gradientEdges[y*size+x] = math.Sqrt(gx*gx + gy*gy)
		}
	}
	
	// Normalize gradient edges
	maxGradient := 0.0
	for i := 0; i < n; i++ {
		if gradientEdges[i] > maxGradient {
			maxGradient = gradientEdges[i]
		}
	}
	if maxGradient > 0 {
		for i := 0; i < n; i++ {
			gradientEdges[i] /= maxGradient
		}
	}

	// Convert to frequency domain
	fft := t.fft2D(data, size)
	fmt.Println("   FFT transform completed")
	
	// Initialize edge map and orientation map
	edges := make([]float64, n)
	orientations := make([]float64, n)
	
	// Store shearlet coefficients for each scale and direction
	coeffsMap := make([][][]complex128, t.scales)
	totalFilters := 0
	for scale := 0; scale < t.scales; scale++ {
		maxShear := int(math.Pow(2, float64(scale)))
		numShears := 2*maxShear + 1
		totalFilters += numShears
	}
	
	fmt.Printf("   Applying %d shearlet filters across %d scales...\n", totalFilters, t.scales)
	processedFilters := 0
	
	for scale := 0; scale < t.scales; scale++ {
		maxShear := int(math.Pow(2, float64(scale)))
		numShears := 2*maxShear + 1
		coeffsMap[scale] = make([][]complex128, numShears)
		
		for shearIdx, shear := range t.getShearRange(maxShear) {
			// Apply shearlet filter
			coeffsMap[scale][shearIdx] = t.applyShearletFilter(fft, scale, shear, size)
			
			// Update progress
			processedFilters++
			if processedFilters%5 == 0 || processedFilters == totalFilters {
				fmt.Printf("   Progress: %d%% (%d/%d filters applied)\n", 
					int(float64(processedFilters)/float64(totalFilters)*100),
					processedFilters, totalFilters)
			}
		}
	}
	
	fmt.Println("   Computing edge strengths and orientations...")
	
	// Calculate edge map and orientations
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			pixelIdx := i*size + j
			maxCoeff := 0.0
			maxScale := 0
			maxShearIdx := 0
			
			// Find maximum coefficient across all scales and directions
			for scale := 0; scale < t.scales; scale++ {
				maxShear := int(math.Pow(2, float64(scale)))
				shearRange := t.getShearRange(maxShear)
				
				for shearIdx := 0; shearIdx < len(shearRange); shearIdx++ {
					coeff := cmplx.Abs(coeffsMap[scale][shearIdx][pixelIdx])
					if coeff > maxCoeff {
						maxCoeff = coeff
						maxScale = scale
						maxShearIdx = shearIdx
					}
				}
			}
			
			// Store edge strength - combine shearlet response with gradient magnitude
			// This helps with synthetic images that have sharp edges
			edges[pixelIdx] = 0.5*maxCoeff + 0.5*gradientEdges[pixelIdx]
			
			// Calculate orientation based on the shear parameter with maximum response
			// Following the paper's equation: θ_j(e) = arg max_k |SH(I)(j,k,e)|
			maxShear := int(math.Pow(2, float64(maxScale)))
			shearRange := t.getShearRange(maxShear)
			shear := shearRange[maxShearIdx]
			orientations[pixelIdx] = math.Atan2(float64(shear), 1.0)
		}
	}

	// Normalize edge map
	maxEdge := 0.0
	for i := 0; i < n; i++ {
		if edges[i] > maxEdge {
			maxEdge = edges[i]
		}
	}
	if maxEdge > 0 {
		for i := 0; i < n; i++ {
			edges[i] /= maxEdge
		}
	}

	fmt.Println("   Edge detection completed")
	return EdgeInfo{
		Edges:       edges,
		Orientations: orientations,
	}
}

// getShearRange returns the range of shear parameters for a given maximum shear
func (t *Transform) getShearRange(maxShear int) []int {
	shearRange := make([]int, 2*maxShear+1)
	for i := 0; i <= 2*maxShear; i++ {
		shearRange[i] = i - maxShear
	}
	return shearRange
}

// DetectEdges is a simplified version that only returns the edge map
func (t *Transform) DetectEdges(data []float64) []float64 {
	edgeInfo := t.DetectEdgesWithOrientation(data)
	return edgeInfo.Edges
}

// applyShearletFilter applies a shearlet filter at a specific scale and shear
func (t *Transform) applyShearletFilter(fft []complex128, scale, shear int, size int) []complex128 {
	result := make([]complex128, len(fft))
	
	// Get shearlet generator for this scale
	psi := t.psi[scale]
	psiSize := len(psi)
	genSize := int(math.Sqrt(float64(psiSize)))
	
	// Apply shearing and scaling
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			// Calculate sheared coordinates
			iSheared := i
			jSheared := j + shear*i
			
			idx := i*size + j
			if idx >= len(fft) {
				continue
			}
			
			// Only process if sheared coordinates are within bounds
			if jSheared >= 0 && jSheared < size && iSheared < size {
				// Make sure we respect the generator size
				if genSize == size {
					idxSheared := iSheared*size + jSheared
					if idxSheared < psiSize {
						// Apply filter
						result[idx] = fft[idx] * psi[idxSheared]
					}
				} else {
					// Scale the coordinates to the generator size
					iGen := iSheared * genSize / size
					jGen := jSheared * genSize / size
					if iGen < genSize && jGen < genSize {
						idxGen := iGen*genSize + jGen
						if idxGen < psiSize {
							// Apply filter
							result[idx] = fft[idx] * psi[idxGen]
						}
					}
				}
			}
		}
	}
	
	return result
}

// ApplyEdgePreservedSmoothing implements Algorithm 2 from the paper
// "Edge preserved Kriging interpolation" with mean-median logic
func (t *Transform) ApplyEdgePreservedSmoothing(data []float64) []float64 {
	n := len(data)
	result := make([]float64, n)
	copy(result, data)

	size := int(math.Sqrt(float64(n)))
	if size*size != n {
		return result
	}

	// Get edges and their orientations
	edgeInfo := t.DetectEdgesWithOrientation(data)
	edges := edgeInfo.Edges
	orientations := edgeInfo.Orientations

	// Apply threshold to identify edge pixels
	threshold := 0.2 // Adjustable threshold for edge detection
	edgePixels := make([]bool, n)
	for i := 0; i < n; i++ {
		edgePixels[i] = edges[i] > threshold
	}

	// Process each pixel
	for i := 1; i < size-1; i++ {
		for j := 1; j < size-1; j++ {
			pixelIdx := i*size + j
			
			// Skip non-edge pixels
			if !edgePixels[pixelIdx] {
				continue
			}
			
			// Apply window-based processing as described in Algorithm 2
			// Use a sliding window of 16 edge pixels
			orientations, changed := t.processEdgeWindow(i, j, size, edgePixels, orientations)
			
			// If orientation was changed, adjust pixel values using mean-median logic
			if changed {
				// Apply mean-median logic as described in the paper
				t.applyMeanMedianLogic(result, i, j, size, orientations[pixelIdx])
			}
		}
	}

	return result
}

// processEdgeWindow implements the window-based processing in Algorithm 2
func (t *Transform) processEdgeWindow(x, y, size int, edgePixels []bool, orientations []float64) ([]float64, bool) {
	// Define window size (16 as mentioned in the paper)
	windowSize := 16
	edgePixelIndices := make([]int, 0, windowSize)
	
	// Find 16 edge pixels in clockwise direction
	// This is a simplified approach - in a real implementation, you would
	// use a more sophisticated method to follow edges in clockwise direction
	for i := x - 2; i <= x + 2; i++ {
		for j := y - 2; j <= y + 2; j++ {
			if i < 0 || i >= size || j < 0 || j >= size {
				continue
			}
			
			pixelIdx := i*size + j
			if edgePixels[pixelIdx] {
				edgePixelIndices = append(edgePixelIndices, pixelIdx)
				if len(edgePixelIndices) >= windowSize {
					break
				}
			}
		}
		if len(edgePixelIndices) >= windowSize {
			break
		}
	}
	
	// Skip if not enough edge pixels in window
	if len(edgePixelIndices) < 3 {
		return orientations, false
	}
	
	// Check if orientations change frequently
	changes := 0
	// We don't actually need to store this, just use it directly
	for i := 1; i < len(edgePixelIndices); i++ {
		diff := math.Abs(orientations[edgePixelIndices[i]] - orientations[edgePixelIndices[i-1]])
		if diff > 0.2 { // Threshold for orientation change
			changes++
		}
	}
	
	// If orientations change frequently, replace with mean orientation
	if float64(changes)/float64(len(edgePixelIndices)) > 0.3 {
		// Calculate mean orientation
		sumOrientation := 0.0
		for _, idx := range edgePixelIndices {
			sumOrientation += orientations[idx]
		}
		meanOrientation := sumOrientation / float64(len(edgePixelIndices))
		
		// Replace orientations
		newOrientations := make([]float64, len(orientations))
		copy(newOrientations, orientations)
		
		for _, idx := range edgePixelIndices {
			newOrientations[idx] = meanOrientation
		}
		
		return newOrientations, true
	}
	
	return orientations, false
}

// applyMeanMedianLogic implements the mean-median logic for pixel adjustment
// as described in the paper (Algorithm 2, Steps 4-5)
func (t *Transform) applyMeanMedianLogic(data []float64, x, y, size int, orientation float64) {
	// We don't need to use pixelIdx, so let's remove it
	
	// Get neighboring pixels
	neighbors := make([]float64, 0, 8)
	for i := -1; i <= 1; i++ {
		for j := -1; j <= 1; j++ {
			if i == 0 && j == 0 {
				continue
			}
			
			ni := x + i
			nj := y + j
			
			if ni < 0 || ni >= size || nj < 0 || nj >= size {
				continue
			}
			
			neighbors = append(neighbors, data[ni*size+nj])
		}
	}
	
	// Skip if not enough neighbors
	if len(neighbors) < 6 {
		return
	}
	
	// Determine left and right pixels based on orientation
	// This is a simplification - a more precise implementation would use
	// the orientation to determine the exact left/right division
	leftNeighbors := make([]float64, 0, 3)
	rightNeighbors := make([]float64, 0, 3)
	
	// Basic division based on orientation (simplified)
	if orientation >= -math.Pi/4 && orientation < math.Pi/4 {
		// Horizontal edge, divide top/bottom
		for i := -1; i <= 1; i++ {
			for j := -1; j <= 1; j++ {
				if i == 0 || (i == 0 && j == 0) {
					continue
				}
				
				ni := x + i
				nj := y + j
				
				if ni < 0 || ni >= size || nj < 0 || nj >= size {
					continue
				}
				
				if i < 0 {
					leftNeighbors = append(leftNeighbors, data[ni*size+nj])
				} else {
					rightNeighbors = append(rightNeighbors, data[ni*size+nj])
				}
			}
		}
	} else {
		// Vertical edge, divide left/right
		for i := -1; i <= 1; i++ {
			for j := -1; j <= 1; j++ {
				if j == 0 || (i == 0 && j == 0) {
					continue
				}
				
				ni := x + i
				nj := y + j
				
				if ni < 0 || ni >= size || nj < 0 || nj >= size {
					continue
				}
				
				if j < 0 {
					leftNeighbors = append(leftNeighbors, data[ni*size+nj])
				} else {
					rightNeighbors = append(rightNeighbors, data[ni*size+nj])
				}
			}
		}
	}
	
	// Apply median to left and right sides
	leftMedian := median(leftNeighbors)
	rightMedian := median(rightNeighbors)
	
	// Update neighbor pixels with median values
	// This implements the logic from Algorithm 2, Step 5
	for i := -1; i <= 1; i++ {
		for j := -1; j <= 1; j++ {
			if i == 0 && j == 0 {
				continue
			}
			
			ni := x + i
			nj := y + j
			
			if ni < 0 || ni >= size || nj < 0 || nj >= size {
				continue
			}
			
			// Determine if this is a left or right neighbor
			isLeft := false
			if orientation >= -math.Pi/4 && orientation < math.Pi/4 {
				isLeft = i < 0
			} else {
				isLeft = j < 0
			}
			
			// Update with appropriate median
			if isLeft {
				data[ni*size+nj] = leftMedian
			} else {
				data[ni*size+nj] = rightMedian
			}
		}
	}
}

// median calculates the median value of a slice
func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	
	// Copy values to avoid modifying the original
	sorted := make([]float64, len(values))
	copy(sorted, values)
	
	// Sort values
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	
	// Return median
	if len(sorted)%2 == 0 {
		return (sorted[len(sorted)/2-1] + sorted[len(sorted)/2]) / 2
	}
	return sorted[len(sorted)/2]
}

// SmoothEdges applies edge-preserving smoothing to the input data
func (t *Transform) SmoothEdges(data []float64, edges []float64) []float64 {
	// Apply edge-preserved smoothing
	return t.ApplyEdgePreservedSmoothing(data)
}

// meyer implements the Meyer auxiliary function used in wavelet construction.
// This function provides a smooth transition between 0 and 1 in the interval [0,1].
// It is used to create smooth partitioning of the frequency domain for the shearlet transform.
//
// Parameters:
//   - t: Input value in the range [0,1]
//
// Returns:
//   - A smoothly varying value that equals 0 at t=0 and 1 at t=1
func meyer(t float64) float64 {
	if t <= 0 {
		return 0
	} else if t >= 1 {
		return 1
	}
	return t*t*(3-2*t)
} 