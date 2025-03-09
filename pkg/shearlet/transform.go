package shearlet

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/cmplx"
	"os"
)

// Transform implements the discrete shearlet transform as described in the paper
// "Fast 3D Volumetric Image Reconstruction from 2D MRI Slices by Parallel Processing".
type Transform struct {
	scales    int
	shears    int
	mother    []complex128
	psi       [][]complex128
	phi       []complex128
	coneParam float64
}

// EdgeInfo holds edge detection information
type EdgeInfo struct {
	Edges        []float64
	Orientations []float64
}

// NewTransform creates a new shearlet transform instance
func NewTransform() *Transform {
	t := &Transform{
		scales:    3, // Reduced number of scales for initial testing
		shears:    8,
		coneParam: 1.0,
	}
	t.initializeGenerators()
	return t
}

// initializeGenerators creates shearlet generators for each scale
func (t *Transform) initializeGenerators() {
	size := 32
	t.psi = make([][]complex128, t.scales)

	t.phi = t.createScalingFunction(size)

	for j := 0; j < t.scales; j++ {
		t.psi[j] = t.createShearletGenerator(size, j)

		// Save the filter as an image for debugging
		t.saveFilterAsImage(t.psi[j], size, fmt.Sprintf("shearlet_filter_scale_%d.png", j))
	}
}

// createScalingFunction generates the scaling function (father wavelet)
func (t *Transform) createScalingFunction(size int) []complex128 {
	phi := make([]complex128, size*size)

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			ω1 := 2 * math.Pi * float64(i-size/2) / float64(size)
			ω2 := 2 * math.Pi * float64(j-size/2) / float64(size)
			r := math.Sqrt(ω1*ω1 + ω2*ω2)

			var value float64
			if r <= math.Pi/4 {
				value = 1
			} else if r <= math.Pi/2 {
				tmp := (4/math.Pi)*r - 1
				value = math.Cos(math.Pi / 2 * meyer(tmp))
			}

			phi[i*size+j] = complex(value, 0)
		}
	}

	return phi
}

func (t *Transform) createShearletGenerator(size int, scale int) []complex128 {
	psi := make([]complex128, size*size)

	a := math.Pow(2, float64(scale)) // parabolic scaling  (Corrected: Now just scaling, not 2^(-2*scale))
	s := 1.0                         // Fixed shear parameter.

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {

			// Frequency domain coordinates, centered at 0
			w1 := float64(j-size/2) / float64(size/2) // Normalized to [-1, 1]
			w2 := float64(i-size/2) / float64(size/2) // Normalized to [-1, 1]

			// Shear transform
			w1s := w1 + s*w2

			// Radial component (Mexican hat wavelet)
			radial := t.mexicanHat(math.Sqrt(w1s*w1s + w2*w2))

			// Angular component (directional)
			angular := math.Exp(-0.5 * (w2 * w2) / a) // Corrected for a scaling,  Gaussian directional window

			// Combined filter response
			value := radial * angular

			psi[i*size+j] = complex(value, 0)
		}
	}
	return psi
}

// mexicanHat is a bandpass filter
func (t *Transform) mexicanHat(radius float64) float64 {
	// Constants for Mexican Hat Wavelet
	const sigma = 0.5
	var norm = 1.0 / (math.Sqrt(2*math.Pi) * math.Pow(sigma, 5))

	// Calculate wavelet value
	r2 := radius * radius
	val := (1 - r2/(2*sigma*sigma)) * math.Exp(-r2/(2*sigma*sigma))

	return norm * val
}

// saveFilterAsImage saves a shearlet filter as a grayscale PNG image for visualization
func (t *Transform) saveFilterAsImage(filter []complex128, size int, filename string) error {
	img := image.NewGray(image.Rect(0, 0, size, size))
	maxVal := 0.0

	// Find the maximum absolute value for normalization
	for _, c := range filter {
		absVal := cmplx.Abs(c)
		if absVal > maxVal {
			maxVal = absVal
		}
	}

	// Normalize and set pixel values
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			absVal := cmplx.Abs(filter[i*size+j])
			grayVal := uint8((absVal / maxVal) * 255) // Normalize to 0-255
			img.SetGray(j, i, color.Gray{Y: grayVal})
		}
	}

	// Create file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Encode to PNG
	return png.Encode(file, img)
}

// DetectEdgesWithOrientation applies the shearlet transform to detect edges and their orientations
func (t *Transform) DetectEdgesWithOrientation(data []float64) EdgeInfo {
	n := len(data)
	size := int(math.Sqrt(float64(n)))
	if size*size != n {
		return EdgeInfo{
			Edges:        make([]float64, n),
			Orientations: make([]float64, n),
		}
	}

	fmt.Printf("   Detecting edges in %dx%d image...\n", size, size)

	edges := make([]float64, n)
	orientations := make([]float64, n)

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
			coeffsMap[scale][shearIdx] = t.applyShearletFilter(data, scale, shear, size)

			processedFilters++
			if processedFilters%5 == 0 || processedFilters == totalFilters {
				fmt.Printf("   Progress: %d%% (%d/%d filters applied)\n",
					int(float64(processedFilters)/float64(totalFilters)*100),
					processedFilters, totalFilters)
			}
		}
	}

	fmt.Println("   Computing edge strengths and orientations...")

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			pixelIdx := i*size + j
			maxCoeff := 0.0
			maxScale := 0
			maxShearIdx := 0

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

			// Store edge strength - Removed Gradient Magnitude
			edges[pixelIdx] = maxCoeff //0.5*maxCoeff + 0.5*gradientEdges[pixelIdx]

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

	if maxEdge > 0 { // Added Zero check.
		for i := 0; i < n; i++ {
			edges[i] /= maxEdge
		}
	}

	fmt.Println("   Edge detection completed")
	return EdgeInfo{
		Edges:        edges,
		Orientations: orientations,
	}
}

// DetectEdges is a simplified version that only returns the edge map
func (t *Transform) DetectEdges(data []float64) []float64 {
	edgeInfo := t.DetectEdgesWithOrientation(data)
	return edgeInfo.Edges
}

// DetectEdgesWithThreshold detects edges using a custom threshold value
func (t *Transform) DetectEdgesWithThreshold(data []float64, threshold float64) []float64 {
	n := len(data)
	edges := make([]float64, n)

	edgeInfo := t.DetectEdgesWithOrientation(data)

	// Apply the custom threshold
	for i := 0; i < n; i++ {
		if edgeInfo.Edges[i] > threshold {
			edges[i] = 1.0 // Edge detected
		} else {
			edges[i] = 0.0 // No edge
		}
	}

	return edges
}

// getShearRange returns the range of shear parameters for a given maximum shear
func (t *Transform) getShearRange(maxShear int) []int {
	shearRange := make([]int, 2*maxShear+1)
	for i := 0; i <= 2*maxShear; i++ {
		shearRange[i] = i - maxShear
	}
	return shearRange
}

// applyShearletFilter applies a shearlet filter at a specific scale and shear
func (t *Transform) applyShearletFilter(data []float64, scale, shear int, size int) []complex128 {
	result := make([]complex128, len(data))

	psi := t.psi[scale]
	psiSize := len(psi)
	genSize := int(math.Sqrt(float64(psiSize)))

	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			iSheared := i
			jSheared := j + shear*i

			idx := i*size + j
			if idx >= len(data) {
				continue
			}

			if jSheared >= 0 && jSheared < size && iSheared < size {
				if genSize == size {
					idxSheared := iSheared*size + jSheared
					if idxSheared < psiSize {
						result[idx] = complex(data[idx], 0) * psi[idxSheared]
					}
				} else {
					iGen := iSheared * genSize / size
					jGen := jSheared * genSize / size
					if iGen < genSize && jGen < genSize {
						idxGen := iGen*genSize + jGen
						if idxGen < psiSize {
							result[idx] = complex(data[idx], 0) * psi[idxGen]
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

	edgeInfo := t.DetectEdgesWithOrientation(data)
	edges := edgeInfo.Edges
	orientations := edgeInfo.Orientations

	threshold := 0.2 // Adjustable threshold for edge detection
	edgePixels := make([]bool, n)
	for i := 0; i < n; i++ {
		edgePixels[i] = edges[i] > threshold
	}

	for i := 1; i < size-1; i++ {
		for j := 1; j < size-1; j++ {
			pixelIdx := i*size + j

			if !edgePixels[pixelIdx] {
				continue
			}

			orientations, changed := t.processEdgeWindow(i, j, size, edgePixels, orientations)

			if changed {
				t.applyMeanMedianLogic(result, i, j, size, orientations[pixelIdx])
			}
		}
	}

	return result
}

// processEdgeWindow implements the window-based processing in Algorithm 2
func (t *Transform) processEdgeWindow(x, y, size int, edgePixels []bool, orientations []float64) ([]float64, bool) {
	windowSize := 16
	edgePixelIndices := make([]int, 0, windowSize)

	for i := x - 2; i <= x+2; i++ {
		for j := y - 2; j <= y+2; j++ {
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

	if len(edgePixelIndices) < 3 {
		return orientations, false
	}

	changes := 0
	for i := 1; i < len(edgePixelIndices); i++ {
		diff := math.Abs(orientations[edgePixelIndices[i]] - orientations[edgePixelIndices[i-1]])
		if diff > 0.2 { // Threshold for orientation change
			changes++
		}
	}

	if float64(changes)/float64(len(edgePixelIndices)) > 0.3 {
		sumOrientation := 0.0
		for _, idx := range edgePixelIndices {
			sumOrientation += orientations[idx]
		}
		meanOrientation := sumOrientation / float64(len(edgePixelIndices))

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
func (t *Transform) applyMeanMedianLogic(data []float64, x, y, size int, orientation float64) {

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

	if len(neighbors) < 6 {
		return
	}

	leftNeighbors := make([]float64, 0, 3)
	rightNeighbors := make([]float64, 0, 3)

	if orientation >= -math.Pi/4 && orientation < math.Pi/4 {
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

	leftMedian := median(leftNeighbors)
	rightMedian := median(rightNeighbors)

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

			isLeft := false
			if orientation >= -math.Pi/4 && orientation < math.Pi/4 {
				isLeft = i < 0
			} else {
				isLeft = j < 0
			}

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

	sorted := make([]float64, len(values))
	copy(sorted, values)

	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	if len(sorted)%2 == 0 {
		return (sorted[len(sorted)/2-1] + sorted[len(sorted)/2]) / 2
	}
	return sorted[len(sorted)/2]
}

// SmoothEdges applies edge-preserving smoothing to the input data
func (t *Transform) SmoothEdges(data []float64, edges []float64) []float64 {
	// Removed Edge Smoothing
	//return t.ApplyEdgePreservedSmoothing(data)
	return data
}

// meyer implements the Meyer auxiliary function used in wavelet construction.
func meyer(t float64) float64 {
	if t < 0 {
		return 0
	} else if t > 1 {
		return 1
	}
	return t * t * (3 - 2*t)
}
