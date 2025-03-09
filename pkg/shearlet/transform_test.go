package shearlet

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// TestNewTransform ensures that a new transform is correctly initialized
// with default parameters
func TestNewTransform(t *testing.T) {
	transform := NewDefaultTransform()
	
	// Verify default parameters
	if transform.scales != 3 {
		t.Errorf("Expected scales=3, got %d", transform.scales)
	}
	
	if transform.shears != 8 {
		t.Errorf("Expected shears=8, got %d", transform.shears)
	}
	
	if transform.coneParam != 1.0 {
		t.Errorf("Expected coneParam=1.0, got %f", transform.coneParam)
	}
	
	// Check if generators are initialized
	if transform.psi == nil {
		t.Errorf("Shearlet generators (psi) not initialized")
	}
	
	if transform.phi == nil {
		t.Errorf("Scaling function (phi) not initialized")
	}
	
	// Check if we have the correct number of generators (one per scale)
	if len(transform.psi) != transform.scales {
		t.Errorf("Expected %d shearlet generators, got %d", transform.scales, len(transform.psi))
	}
}

// TestGetShearRange verifies the shear parameter range calculation
func TestGetShearRange(t *testing.T) {
	transform := NewDefaultTransform()
	
	// Test shear range for maxShear = 1
	range1 := transform.getShearRange(1)
	expected1 := []int{-1, 0, 1}
	
	if len(range1) != len(expected1) {
		t.Errorf("Expected shear range of length %d, got %d", len(expected1), len(range1))
	}
	
	for i, val := range range1 {
		if val != expected1[i] {
			t.Errorf("Expected shear range[%d]=%d, got %d", i, expected1[i], val)
		}
	}
	
	// Test shear range for maxShear = 2
	range2 := transform.getShearRange(2)
	expected2 := []int{-2, -1, 0, 1, 2}
	
	if len(range2) != len(expected2) {
		t.Errorf("Expected shear range of length %d, got %d", len(expected2), len(range2))
	}
	
	for i, val := range range2 {
		if val != expected2[i] {
			t.Errorf("Expected shear range[%d]=%d, got %d", i, expected2[i], val)
		}
	}
}

// TestDetectEdges verifies edge detection on a simple test pattern
func TestDetectEdges(t *testing.T) {
	transform := NewDefaultTransform()
	
	// Create a simple 32x32 test image with a vertical edge
	// (using 32x32 to match the internal size used by the transform)
	size := 32
	testImage := make([]float64, size*size)
	
	// Create a vertical edge in the middle with high contrast
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if x >= size/2 {
				testImage[y*size+x] = 1.0
			} else {
				testImage[y*size+x] = 0.0
			}
		}
	}
	
	// Detect edges
	edges := transform.DetectEdges(testImage)
	
	// Verify that edges were detected
	if len(edges) != size*size {
		t.Fatalf("Expected edge map of size %d, got %d", size*size, len(edges))
	}
	
	// Check for edge response near the vertical edge
	hasEdgeResponse := false
	middleY := size / 2
	
	// Check a few columns around the edge
	for x := size/2 - 2; x <= size/2 + 2; x++ {
		if x >= 0 && x < size {
			if edges[middleY*size+x] > 0.3 {  // Increased threshold since we expect stronger response
				hasEdgeResponse = true
				break
			}
		}
	}
	
	if !hasEdgeResponse {
		// Print edge values around the expected edge location for debugging
		fmt.Println("Edge values around the vertical edge:")
		for x := size/2 - 3; x <= size/2 + 3; x++ {
			if x >= 0 && x < size {
				fmt.Printf("  Edge at (%d,%d): %.3f\n", x, middleY, edges[middleY*size+x])
			}
		}
		t.Errorf("No edge response detected near the vertical edge")
	}
}

// TestDetectEdgesWithOrientation verifies edge detection and orientation
// on a simple test pattern
func TestDetectEdgesWithOrientation(t *testing.T) {
	transform := NewDefaultTransform()
	
	// Create a simple 32x32 test image with a vertical edge
	// (using 32x32 to match the internal size used by the transform)
	size := 32
	testImage := make([]float64, size*size)
	
	// Create a vertical edge in the middle
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if x >= size/2 {
				testImage[y*size+x] = 1.0
			} else {
				testImage[y*size+x] = 0.0
			}
		}
	}
	
	// Detect edges and their orientations
	edgeInfo := transform.DetectEdgesWithOrientation(testImage)
	
	// Verify we have some edges and orientations
	hasEdges := false
	for _, val := range edgeInfo.Edges {
		if val > 0.3 {
			hasEdges = true
			break
		}
	}
	
	if !hasEdges {
		// Print edge values around the expected edge location for debugging
		middleY := size / 2
		fmt.Println("Edge values around the vertical edge:")
		for x := size/2 - 3; x <= size/2 + 3; x++ {
			if x >= 0 && x < size {
				fmt.Printf("  Edge at (%d,%d): %.3f\n", x, middleY, edgeInfo.Edges[middleY*size+x])
			}
		}
		t.Errorf("No edge response detected in the entire image")
		return
	}
	
	// Check that orientations are present and valid
	for i, orientation := range edgeInfo.Orientations {
		// Orientations should be between -π and +π
		if orientation < -math.Pi || orientation > math.Pi {
			t.Errorf("Invalid orientation at index %d: %f", i, orientation)
		}
	}
	
	// For a vertical edge, orientations near the edge should be close to 0
	// (since atan2(0, 1) = 0 for a vertical edge)
	middleY := size / 2
	middleX := size / 2
	
	// Check orientation at the edge
	edgeOrientation := edgeInfo.Orientations[middleY*size+middleX]
	if math.Abs(edgeOrientation) > math.Pi/4 {
		t.Errorf("Expected orientation near 0 for vertical edge, got %f", edgeOrientation)
	}
}

// TestApplyEdgePreservedSmoothing verifies the edge-preserving smoothing algorithm
func TestApplyEdgePreservedSmoothing(t *testing.T) {
	transform := NewDefaultTransform()
	
	// Create a test image with a vertical edge and some noise
	size := 32
	testImage := make([]float64, size*size)
	
	// Create a vertical edge in the middle
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			// Base image with vertical edge
			if x >= size/2 {
				testImage[y*size+x] = 1.0
			} else {
				testImage[y*size+x] = 0.0
			}
			
			// Add some noise
			testImage[y*size+x] += 0.1 * (math.Sin(float64(x*y)/10) - 0.5)
		}
	}
	
	// Apply edge-preserved smoothing
	smoothedImage := transform.ApplyEdgePreservedSmoothing(testImage)
	
	// Verify the result has the same size
	if len(smoothedImage) != len(testImage) {
		t.Fatalf("Expected smoothed image of size %d, got %d", len(testImage), len(smoothedImage))
	}
	
	// Check that the edge is preserved
	// The edge should still be sharp after smoothing
	
	// Calculate the average gradient magnitude at the edge
	edgeGradient := 0.0
	for y := 1; y < size-1; y++ {
		// Gradient at the edge (x = size/2)
		gradient := math.Abs(smoothedImage[y*size+size/2] - smoothedImage[y*size+size/2-1])
		edgeGradient += gradient
	}
	edgeGradient /= float64(size - 2)
	
	// Calculate the average gradient magnitude away from the edge
	nonEdgeGradient := 0.0
	for y := 1; y < size-1; y++ {
		// Gradient away from the edge (x = size/4)
		gradient := math.Abs(smoothedImage[y*size+size/4] - smoothedImage[y*size+size/4-1])
		nonEdgeGradient += gradient
	}
	nonEdgeGradient /= float64(size - 2)
	
	// The gradient at the edge should be significantly higher than away from the edge
	if edgeGradient <= nonEdgeGradient*2 {
		t.Errorf("Edge not preserved: edge gradient (%.3f) should be significantly higher than non-edge gradient (%.3f)",
			edgeGradient, nonEdgeGradient)
	}
	
	// Check that noise has been reduced
	// Calculate the variance in flat regions before and after smoothing
	leftVarianceBefore := calculateVariance(testImage, 0, size/4, 0, size)
	leftVarianceAfter := calculateVariance(smoothedImage, 0, size/4, 0, size)
	
	rightVarianceBefore := calculateVariance(testImage, 3*size/4, size, 0, size)
	rightVarianceAfter := calculateVariance(smoothedImage, 3*size/4, size, 0, size)
	
	// Variance should be reduced after smoothing in at least one of the regions
	// or the total variance should be reduced
	totalVarianceBefore := leftVarianceBefore + rightVarianceBefore
	totalVarianceAfter := leftVarianceAfter + rightVarianceAfter
	
	if totalVarianceAfter >= totalVarianceBefore && 
	   leftVarianceAfter >= leftVarianceBefore && 
	   rightVarianceAfter >= rightVarianceBefore {
		t.Errorf("Noise not reduced: variance before (%.6f, %.6f), after (%.6f, %.6f)",
			leftVarianceBefore, rightVarianceBefore, leftVarianceAfter, rightVarianceAfter)
	}
}

// calculateVariance computes the variance of pixel values in a region
func calculateVariance(data []float64, x1, x2, y1, y2 int) float64 {
	size := int(math.Sqrt(float64(len(data))))
	
	// Calculate mean
	sum := 0.0
	count := 0
	for y := y1; y < y2; y++ {
		for x := x1; x < x2; x++ {
			if y*size+x < len(data) {
				sum += data[y*size+x]
				count++
			}
		}
	}
	mean := sum / float64(count)
	
	// Calculate variance
	variance := 0.0
	for y := y1; y < y2; y++ {
		for x := x1; x < x2; x++ {
			if y*size+x < len(data) {
				diff := data[y*size+x] - mean
				variance += diff * diff
			}
		}
	}
	return variance / float64(count)
}

// TestMeyer verifies the Meyer window function implementation
func TestMeyer(t *testing.T) {
	// Test values outside [0,1] range
	if meyer(-0.5) != 0.0 {
		t.Errorf("Expected meyer(-0.5)=0, got %f", meyer(-0.5))
	}
	
	if meyer(1.5) != 1.0 {
		t.Errorf("Expected meyer(1.5)=1, got %f", meyer(1.5))
	}
	
	// Test values inside range with known results
	testCases := []struct {
		input    float64
		expected float64
	}{
		{0.0, 0.0},
		{0.25, 0.25 * 0.25 * (3 - 2 * 0.25)},
		{0.5, 0.5 * 0.5 * (3 - 2 * 0.5)}, // = 0.5*0.5*2 = 0.5
		{0.75, 0.75 * 0.75 * (3 - 2 * 0.75)},
		{1.0, 1.0},
	}
	
	for _, tc := range testCases {
		result := meyer(tc.input)
		if math.Abs(result-tc.expected) > 1e-6 {
			t.Errorf("Expected meyer(%f)=%f, got %f", tc.input, tc.expected, result)
		}
	}
}

// TestFFT1D verifies the 1D FFT implementation
func TestFFT1D(t *testing.T) {
	// Create a simple test signal
	n := 8
	x := make([]complex128, n)
	for i := 0; i < n; i++ {
		x[i] = complex(float64(i), 0)
	}
	
	// Compute FFT
	y := complexFFT(x)
	
	// Verify basic properties
	if len(y) != n {
		t.Errorf("Expected FFT output length %d, got %d", n, len(y))
	}
	
	// DC component should be sum of input
	sum := complex(0, 0)
	for _, v := range x {
		sum += v
	}
	if math.Abs(real(y[0])-real(sum)) > 1e-10 || math.Abs(imag(y[0])-imag(sum)) > 1e-10 {
		t.Errorf("Expected DC component %v, got %v", sum, y[0])
	}
	
	// Nyquist component should be alternating sum
	altSum := complex(0, 0)
	for i, v := range x {
		if i%2 == 0 {
			altSum += v
		} else {
			altSum -= v
		}
	}
	if math.Abs(real(y[n/2])-real(altSum)) > 1e-10 || math.Abs(imag(y[n/2])-imag(altSum)) > 1e-10 {
		t.Errorf("Expected Nyquist component %v, got %v", altSum, y[n/2])
	}
}

// TestFFT2D verifies the 2D FFT implementation using a simple test case
func TestFFT2D(t *testing.T) {
	transform := NewDefaultTransform()
	
	// Create a 2x2 test image with impulse at origin
	testImage := []float64{1, 0, 0, 0}
	
	// 2D FFT of impulse should be constant in frequency domain
	result := transform.fft2D(testImage, 2)
	
	if len(result) != 4 {
		t.Fatalf("Expected FFT result of length 4, got %d", len(result))
	}
	
	// All values should be approximately equal to 1
	for i, val := range result {
		if math.Abs(cmplx.Abs(val)-1.0) > 1e-6 {
			t.Errorf("FFT[%d]: expected magnitude close to 1.0, got %v", i, cmplx.Abs(val))
		}
	}
}