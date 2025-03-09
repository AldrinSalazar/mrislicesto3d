package interpolation

import (
	"fmt"
	"math"
	"testing"
)

// TestNewKriging verifies that a kriging interpolator is created with correct parameters
func TestNewKriging(t *testing.T) {
	// Create test data
	width := 4
	height := 4
	data := createTestData(width, height, 2)
	sliceGap := 2.0

	// Create kriging interpolator
	k := NewKriging(data, sliceGap)

	// Verify it was created correctly
	if k == nil {
		t.Fatal("Failed to create kriging interpolator")
	}

	if k.sliceGap != sliceGap {
		t.Errorf("Expected slice gap %f, got %f", sliceGap, k.sliceGap)
	}

	if k.width != width {
		t.Errorf("Expected width %d, got %d", width, k.width)
	}

	if k.height != height {
		t.Errorf("Expected height %d, got %d", height, k.height)
	}

	// Check that data points were set up
	if len(k.dataPoints) != len(data) {
		t.Errorf("Expected %d data points, got %d", len(data), len(k.dataPoints))
	}
}

// TestVariogramModels verifies the three variogram models (Spherical, Exponential, Gaussian)
func TestVariogramModels(t *testing.T) {
	// Create a simple kriging object
	data := createTestData(4, 4, 1)
	k := NewKriging(data, 1.0)

	// Test parameters
	testParams := []KrigingParams{
		{Range: 10.0, Sill: 1.0, Nugget: 0.1, Model: Spherical},
		{Range: 10.0, Sill: 1.0, Nugget: 0.1, Model: Exponential},
		{Range: 10.0, Sill: 1.0, Nugget: 0.1, Model: Gaussian},
	}

	// Test distances
	distances := []float64{0.0, 5.0, 10.0, 20.0}

	// Expected properties for each model
	for _, params := range testParams {
		k.params = params

		for _, h := range distances {
			gamma := k.variogram(h, params)
			
			// Common checks for all models
			if h == 0 {
				if gamma != 0 {
					t.Errorf("Variogram at h=0 should be 0, got %f", gamma)
				}
				continue
			}

			// Nugget check
			if gamma < params.Nugget {
				t.Errorf("Variogram should include nugget effect (%f), got %f", 
					params.Nugget, gamma)
			}

			// Sill check for large distances
			if h >= 2*params.Range {
				expectedSill := params.Nugget + params.Sill
				if math.Abs(gamma-expectedSill) > 0.01 {
					t.Errorf("Variogram at large distance should approach sill+nugget (%f), got %f", 
						expectedSill, gamma)
				}
			}

			// Model-specific checks
			switch params.Model {
			case Spherical:
				if h >= params.Range && math.Abs(gamma-(params.Nugget+params.Sill)) > 0.01 {
					t.Errorf("Spherical variogram should reach sill at range=%f", params.Range)
				}
			case Exponential:
				// Exponential model reaches ~95% of sill at 3*range
				if h >= 3*params.Range {
					if gamma < 0.95*(params.Nugget+params.Sill) {
						t.Errorf("Exponential variogram should approach sill at 3*range")
					}
				}
			case Gaussian:
				// Gaussian model reaches ~95% of sill at sqrt(3)*range
				if h >= math.Sqrt(3)*params.Range {
					if gamma < 0.95*(params.Nugget+params.Sill) {
						t.Errorf("Gaussian variogram should approach sill at sqrt(3)*range")
					}
				}
			}
		}
	}
}

// TestDistance3D verifies the distance calculation with anisotropy
func TestDistance3D(t *testing.T) {
	// Create a simple kriging object with anisotropy
	data := createTestData(4, 4, 1)
	k := NewKriging(data, 1.0)
	
	// Set anisotropy
	k.params.Anisotropy.Ratio = 0.5    // y-axis distances are halved
	k.params.Anisotropy.Direction = 0.0 // Aligned with coordinate system
	
	testCases := []struct {
		p1, p2   Point3D
		expected float64
	}{
		// Points along x-axis
		{Point3D{0, 0, 0}, Point3D{3, 0, 0}, 3.0},
		// Points along y-axis (should be scaled by ratio)
		{Point3D{0, 0, 0}, Point3D{0, 2, 0}, 1.0}, // 2 * 0.5 = 1.0
		// Points along z-axis
		{Point3D{0, 0, 0}, Point3D{0, 0, 4}, 4.0},
		// Diagonal points
		{Point3D{0, 0, 0}, Point3D{3, 4, 0}, math.Sqrt(9 + 4)}, // 3^2 + (4*0.5)^2 = 9 + 4 = 13
	}
	
	for i, tc := range testCases {
		result := k.calculateDistance3D(tc.p1, tc.p2)
		if math.Abs(result-tc.expected) > 0.001 {
			t.Errorf("Case %d: Expected distance %.3f, got %.3f", i, tc.expected, result)
		}
	}
	
	// Test with rotated anisotropy
	k.params.Anisotropy.Direction = math.Pi/4 // 45 degrees
	
	// For points along the 45 degree line, there should be no anisotropy effect
	// For points perpendicular to that line, anisotropy should have full effect
	rotatedCases := []struct {
		p1, p2   Point3D
		expected float64
	}{
		// Points along 45 degrees
		{Point3D{0, 0, 0}, Point3D{1, 1, 0}, math.Sqrt(2)},
		// Points along 135 degrees (perpendicular to anisotropy axis)
		{Point3D{0, 0, 0}, Point3D{-1, 1, 0}, math.Sqrt(1 + 0.25)}, // sqrt(1 + (1*0.5)^2)
	}
	
	for i, tc := range rotatedCases {
		result := k.calculateDistance3D(tc.p1, tc.p2)
		if math.Abs(result-tc.expected) > 0.001 {
			t.Errorf("Rotated case %d: Expected distance %.3f, got %.3f", i, tc.expected, result)
		}
	}
}

// TestInterpolate verifies the interpolation function
func TestInterpolate(t *testing.T) {
	// Create test data with a simple pattern (gradient along x-axis)
	width := 4
	height := 4
	numSlices := 2
	sliceGap := 2.0
	
	data := make([]float64, width*height*numSlices)
	
	// Create a gradient pattern
	for z := 0; z < numSlices; z++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				data[z*width*height + y*width + x] = float64(x) / float64(width-1)
			}
		}
	}
	
	// Create kriging interpolator
	k := NewKriging(data, sliceGap)
	k.numSlices = numSlices
	
	// Run interpolation
	result, err := k.Interpolate()
	if err != nil {
		t.Fatalf("Interpolation failed: %v", err)
	}
	
	// Calculate expected dimensions
	slicesPerGap := int(sliceGap)
	totalSlices := (numSlices - 1) * slicesPerGap + 1
	expectedSize := width * height * totalSlices
	
	// Verify dimensions
	if len(result) != expectedSize {
		t.Errorf("Expected result size %d, got %d", expectedSize, len(result))
	}
	
	// Check that original slices are preserved at their correct positions
	for z := 0; z < numSlices; z++ {
		outputZ := z * slicesPerGap
		
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				inputIdx := z*width*height + y*width + x
				outputIdx := outputZ*width*height + y*width + x
				
				if data[inputIdx] != result[outputIdx] {
					t.Errorf("Original data at [%d,%d,%d] not preserved, expected %.2f, got %.2f",
						x, y, z, data[inputIdx], result[outputIdx])
				}
			}
		}
	}
	
	// Check that interpolated values are reasonable - all values should maintain gradient pattern
	for z := 0; z < totalSlices; z++ {
		// Skip original slices
		if z % slicesPerGap == 0 {
			continue
		}
		
		for y := 0; y < height; y++ {
			prevValue := -1.0
			
			for x := 0; x < width; x++ {
				idx := z*width*height + y*width + x
				
				// Values should increase along x-axis (we created a gradient)
				if x > 0 && result[idx] <= prevValue {
					t.Errorf("Expected monotonically increasing values along x-axis at z=%d, y=%d", z, y)
				}
				
				// Values should be in the 0-1 range
				if result[idx] < 0 || result[idx] > 1 {
					t.Errorf("Interpolated value %.2f out of expected range [0,1]", result[idx])
				}
				
				prevValue = result[idx]
			}
		}
	}
}

// TestSolveSystem verifies the linear system solver
func TestSolveSystem(t *testing.T) {
	// Create a simple kriging object
	data := createTestData(4, 4, 1)
	k := NewKriging(data, 1.0)
	
	// Simple 3x3 system with known solution:
	// 2x + y + z = 7
	// x + 3y + z = 10
	// x + y + 4z = 15
	// Solution: x=1, y=2, z=3
	matrix := [][]float64{
		{2, 1, 1},
		{1, 3, 1},
		{1, 1, 4},
	}
	
	target := []float64{7, 10, 15}
	
	solution := k.solveSystem(matrix, target)

	fmt.Println("Solution:", solution)
	
	// Check solution values
	expectedSolution := []float64{1, 2, 3}
	for i, val := range solution {
		if math.Abs(val - expectedSolution[i]) > 0.0001 {
			t.Errorf("Incorrect solution value: expected %.4f, got %.4f", expectedSolution[i], val)
		}
	}
	
	// Verify solution by substituting back into original equations
	for i := 0; i < len(matrix); i++ {
		sum := 0.0
		for j := 0; j < len(matrix[i]); j++ {
			sum += matrix[i][j] * solution[j]
		}
		if math.Abs(sum - target[i]) > 0.0001 {
			t.Errorf("Equation %d does not match: expected %.4f, got %.4f", i, target[i], sum)
		}
	}
}

// TestFindNeighbors verifies the neighbor finding algorithm for 3D points
func TestFindNeighbors(t *testing.T) {
	// Create test data
	width := 5
	height := 5
	numSlices := 2
	data := createTestData(width, height, numSlices)
	
	// Create kriging interpolator
	k := NewKriging(data, 1.0)
	k.width = width
	k.height = height
	k.numSlices = numSlices
	
	// Set up test point in the middle
	testPoint := Point3D{X: 2.5, Y: 2.5, Z: 0.5}
	
	// Test with different radii
	testRadii := []int{1, 2, 3}
	expectedCounts := []int{8, 25, 50} // Theoretical max neighbors
	
	for i, radius := range testRadii {
		neighbors := k.findNeighbors(testPoint, radius)
		
		// Check neighbor count - may be less than max due to boundary effects
		if len(neighbors.Points) == 0 {
			t.Errorf("No neighbors found with radius %d", radius)
		}
		
		if len(neighbors.Points) > expectedCounts[i] {
			t.Errorf("Too many neighbors found: expected max %d, got %d", 
				expectedCounts[i], len(neighbors.Points))
		}
		
		// Check that values and points match
		if len(neighbors.Points) != len(neighbors.Values) {
			t.Errorf("Mismatch between points (%d) and values (%d)", 
				len(neighbors.Points), len(neighbors.Values))
		}
		
		// Check that all points are within the radius
		for j, point := range neighbors.Points {
			distance := math.Sqrt(
				math.Pow(point.X-testPoint.X, 2) + 
				math.Pow(point.Y-testPoint.Y, 2) + 
				math.Pow(point.Z-testPoint.Z, 2))
			
			// Allow some margin due to rounding in coordinate conversions
			maxDist := float64(radius) * 1.5
			if distance > maxDist {
				t.Errorf("Neighbor %d is too far: distance %.2f > max %.2f", j, distance, maxDist)
			}
		}
	}
}

// BenchmarkKrigingInterpolate benchmarks the kriging interpolation performance
func BenchmarkKrigingInterpolate(b *testing.B) {
	// Create test data
	width := 16
	height := 16
	data := createTestData(width, height, 2)
	
	// Reset timer before the actual benchmark
	b.ResetTimer()
	
	// Run the benchmark
	for i := 0; i < b.N; i++ {
		k := NewKriging(data, 2.0)
		_, err := k.Interpolate()
		if err != nil {
			b.Fatalf("Interpolation failed: %v", err)
		}
	}
}

// BenchmarkSolveSystem benchmarks the matrix solving performance
func BenchmarkSolveSystem(b *testing.B) {
	// Create a test matrix and target vector
	n := 20
	matrix := make([][]float64, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i == j {
				matrix[i][j] = 2.0 // Diagonal
			} else if i == j+1 || i+1 == j {
				matrix[i][j] = 1.0 // Off-diagonal
			}
		}
	}
	
	target := make([]float64, n)
	for i := 0; i < n; i++ {
		target[i] = float64(i + 1)
	}
	
	k := &Kriging{}
	
	// Reset timer before the actual benchmark
	b.ResetTimer()
	
	// Run the benchmark
	for i := 0; i < b.N; i++ {
		k.solveSystem(matrix, target)
	}
}

// Helper functions for tests

// createTestData creates a test dataset with the specified dimensions
func createTestData(width, height, numSlices int) []float64 {
	data := make([]float64, width*height*numSlices)
	
	for z := 0; z < numSlices; z++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				// Create a simple pattern (gradient)
				idx := z*width*height + y*width + x
				data[idx] = float64(x + y + z) / float64(width+height+numSlices-3)
			}
		}
	}
	
	return data
}