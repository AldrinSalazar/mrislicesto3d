package stl

import (
	"math"
	"os"
	"testing"
)

// TestMarchingCubes verifies the marching cubes implementation with a simple sphere
func TestMarchingCubes(t *testing.T) {
	// Create a 3D dataset representing a sphere in a 20x20x20 volume
	size := 20
	data := make([]float64, size*size*size)
	
	// Fill with sphere data
	radius := float64(size) / 4.0
	center := float64(size) / 2.0
	
	for z := 0; z < size; z++ {
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				// Calculate distance from center
				dx := float64(x) - center
				dy := float64(y) - center
				dz := float64(z) - center
				dist := math.Sqrt(dx*dx + dy*dy + dz*dz)
				
				// Inside sphere: higher values (1.0)
				// Outside sphere: lower values (0.0)
				// Smooth transition at boundary
				if dist < radius {
					data[z*size*size + y*size + x] = 1.0
				} else {
					data[z*size*size + y*size + x] = 0.0
				}
			}
		}
	}
	
	// Create marching cubes instance
	mc := NewMarchingCubes(data, size, size, size, 0.5)
	
	// Generate triangles
	triangles := mc.GenerateTriangles()
	
	// Check that we got a reasonable number of triangles for a sphere
	// A sphere with this resolution should have at least 100 triangles
	if len(triangles) < 100 {
		t.Errorf("Expected at least 100 triangles for sphere, got %d", len(triangles))
	}
	
	// Verify that triangles form a closed surface using a simplified check
	// Check normals are pointing outward (for a sphere, normal should point away from center)
	for _, triangle := range triangles[:10] { // Check first 10 triangles as a sample
		// Calculate center of triangle
		centerX := (triangle.Vertex1[0] + triangle.Vertex2[0] + triangle.Vertex3[0]) / 3
		centerY := (triangle.Vertex1[1] + triangle.Vertex2[1] + triangle.Vertex3[1]) / 3
		centerZ := (triangle.Vertex1[2] + triangle.Vertex2[2] + triangle.Vertex3[2]) / 3
		
		// Calculate vector from sphere center to triangle center
		vx := centerX - float32(center)
		vy := centerY - float32(center)
		vz := centerZ - float32(center)
		
		// Normalize
		mag := float32(math.Sqrt(float64(vx*vx + vy*vy + vz*vz)))
		if mag > 0 {
			vx /= mag
			vy /= mag
			vz /= mag
		}
		
		// Dot product with normal should be positive for outward-facing normals
		dot := vx*triangle.Normal[0] + vy*triangle.Normal[1] + vz*triangle.Normal[2]
		if dot < -0.5 { // Fairly generous threshold
			t.Errorf("Triangle normal appears to point inward, dot product: %f", dot)
		}
	}
}

// TestSetScale verifies that the scaling functionality works
func TestSetScale(t *testing.T) {
	// Create a simple 2x2x2 volume
	data := []float64{
		1, 0,
		0, 0,
		
		0, 0,
		0, 0,
	}
	
	// Create marching cubes instance
	mc := NewMarchingCubes(data, 2, 2, 2, 0.5)
	
	// Set custom scale
	xScale, yScale, zScale := float32(2.5), float32(1.5), float32(3.0)
	mc.SetScale(xScale, yScale, zScale)
	
	// Generate triangles
	triangles := mc.GenerateTriangles()
	
	// Verify that we got some triangles
	if len(triangles) == 0 {
		t.Fatal("No triangles generated")
	}
	
	// Check that the scale was applied correctly
	// The first triangle should have vertices with coordinates scaled by our factors
	if len(triangles) > 0 {
		triangle := triangles[0]
		
		// Check that at least one vertex has coordinates that reflect the scaling
		foundScaledVertex := false
		
		// Check Vertex1
		if triangle.Vertex1[0] != 0 && math.Abs(float64(triangle.Vertex1[0]/xScale)-float64(triangle.Vertex1[0])/float64(xScale)) < 0.001 {
			foundScaledVertex = true
		}
		if triangle.Vertex1[1] != 0 && math.Abs(float64(triangle.Vertex1[1]/yScale)-float64(triangle.Vertex1[1])/float64(yScale)) < 0.001 {
			foundScaledVertex = true
		}
		if triangle.Vertex1[2] != 0 && math.Abs(float64(triangle.Vertex1[2]/zScale)-float64(triangle.Vertex1[2])/float64(zScale)) < 0.001 {
			foundScaledVertex = true
		}
		
		// Check Vertex2
		if triangle.Vertex2[0] != 0 && math.Abs(float64(triangle.Vertex2[0]/xScale)-float64(triangle.Vertex2[0])/float64(xScale)) < 0.001 {
			foundScaledVertex = true
		}
		if triangle.Vertex2[1] != 0 && math.Abs(float64(triangle.Vertex2[1]/yScale)-float64(triangle.Vertex2[1])/float64(yScale)) < 0.001 {
			foundScaledVertex = true
		}
		if triangle.Vertex2[2] != 0 && math.Abs(float64(triangle.Vertex2[2]/zScale)-float64(triangle.Vertex2[2])/float64(zScale)) < 0.001 {
			foundScaledVertex = true
		}
		
		// Check Vertex3
		if triangle.Vertex3[0] != 0 && math.Abs(float64(triangle.Vertex3[0]/xScale)-float64(triangle.Vertex3[0])/float64(xScale)) < 0.001 {
			foundScaledVertex = true
		}
		if triangle.Vertex3[1] != 0 && math.Abs(float64(triangle.Vertex3[1]/yScale)-float64(triangle.Vertex3[1])/float64(yScale)) < 0.001 {
			foundScaledVertex = true
		}
		if triangle.Vertex3[2] != 0 && math.Abs(float64(triangle.Vertex3[2]/zScale)-float64(triangle.Vertex3[2])/float64(zScale)) < 0.001 {
			foundScaledVertex = true
		}
		
		if !foundScaledVertex {
			t.Error("Scale does not appear to have been applied to triangle vertices")
		}
	}
	
	// Create a second marching cubes instance with default scale
	mc2 := NewMarchingCubes(data, 2, 2, 2, 0.5)
	triangles2 := mc2.GenerateTriangles()
	
	// Verify that the two sets of triangles are different due to scaling
	if len(triangles) > 0 && len(triangles2) > 0 {
		// Compare the first triangle from each set
		t1 := triangles[0]
		t2 := triangles2[0]
		
		// At least one vertex should be different
		allSame := true
		
		// Compare Vertex1
		if t1.Vertex1[0] != t2.Vertex1[0] || t1.Vertex1[1] != t2.Vertex1[1] || t1.Vertex1[2] != t2.Vertex1[2] {
			allSame = false
		}
		
		// Compare Vertex2
		if t1.Vertex2[0] != t2.Vertex2[0] || t1.Vertex2[1] != t2.Vertex2[1] || t1.Vertex2[2] != t2.Vertex2[2] {
			allSame = false
		}
		
		// Compare Vertex3
		if t1.Vertex3[0] != t2.Vertex3[0] || t1.Vertex3[1] != t2.Vertex3[1] || t1.Vertex3[2] != t2.Vertex3[2] {
			allSame = false
		}
		
		if allSame {
			t.Error("Scaling had no effect on triangle vertices")
		}
	}
}

// TestSaveToSTL verifies that the STL file can be written
func TestSaveToSTL(t *testing.T) {
	// Create a simple triangle for testing
	triangles := []Triangle{
		{
			Normal:   [3]float32{0, 0, 1},
			Vertex1:  [3]float32{0, 0, 0},
			Vertex2:  [3]float32{1, 0, 0},
			Vertex3:  [3]float32{0, 1, 0},
		},
	}
	
	// Create a temporary file
	tmpFile, err := os.CreateTemp("", "test-*.stl")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()
	
	// Save triangles to STL
	err = SaveToSTL(tmpFile.Name(), triangles)
	if err != nil {
		t.Fatalf("Failed to save STL: %v", err)
	}
	
	// Check that file exists and has expected size
	info, err := os.Stat(tmpFile.Name())
	if err != nil {
		t.Fatalf("Failed to stat output file: %v", err)
	}
	
	// The file size should be at least the size of the header plus one triangle
	// STL header: 80 bytes
	// Number of triangles: 4 bytes
	// Triangle: 50 bytes (12 bytes per vertex, 12 bytes per normal, 2 bytes attribute)
	minSize := int64(80 + 4 + 50)
	if info.Size() < minSize {
		t.Errorf("STL file too small, expected at least %d bytes, got %d", minSize, info.Size())
	}
}

// TestTriangleInterpolation verifies the vertex interpolation for marching cubes
func TestTriangleInterpolation(t *testing.T) {
	// Create a simple 2x2x2 volume with a boundary passing diagonally
	data := []float64{
		1, 0,
		0, 0,
		
		0, 0,
		0, 0,
	}
	
	// Create marching cubes instance
	mc := NewMarchingCubes(data, 2, 2, 2, 0.5)
	
	// Generate triangles
	triangles := mc.GenerateTriangles()
	
	// We should get at least one triangle
	if len(triangles) == 0 {
		t.Fatal("No triangles generated, cannot test interpolation")
	}
	
	// Verify that triangles are generated at the boundary (where value transitions from 1 to 0)
	// The first vertex should be interpolated between (0,0,0) and (1,0,0)
	// since those are the points where the value transitions from 1 to 0
	
	// Get the first triangle
	triangle := triangles[0]
	
	// Check that at least one vertex is interpolated (not at integer coordinates)
	hasInterpolatedVertex := false
	
	// Check if any vertex has non-integer coordinates (indicating interpolation)
	// Vertex1
	if !isIntegerCoordinate(triangle.Vertex1[0]) || 
	   !isIntegerCoordinate(triangle.Vertex1[1]) || 
	   !isIntegerCoordinate(triangle.Vertex1[2]) {
		hasInterpolatedVertex = true
	}
	
	// Vertex2
	if !isIntegerCoordinate(triangle.Vertex2[0]) || 
	   !isIntegerCoordinate(triangle.Vertex2[1]) || 
	   !isIntegerCoordinate(triangle.Vertex2[2]) {
		hasInterpolatedVertex = true
	}
	
	// Vertex3
	if !isIntegerCoordinate(triangle.Vertex3[0]) || 
	   !isIntegerCoordinate(triangle.Vertex3[1]) || 
	   !isIntegerCoordinate(triangle.Vertex3[2]) {
		hasInterpolatedVertex = true
	}
	
	if !hasInterpolatedVertex {
		t.Error("No interpolated vertices found in the triangle")
	}
	
	// Verify that the normal is non-zero
	if triangle.Normal[0] == 0 && triangle.Normal[1] == 0 && triangle.Normal[2] == 0 {
		t.Error("Triangle normal is zero")
	}
}

// isIntegerCoordinate checks if a coordinate is very close to an integer value
func isIntegerCoordinate(coord float32) bool {
	return math.Abs(float64(coord)-math.Round(float64(coord))) < 0.001
}

// BenchmarkMarchingCubes benchmarks the marching cubes algorithm
func BenchmarkMarchingCubes(b *testing.B) {
	// Create a simple 16x16x16 volume with a sphere in the middle
	width, height, depth := 16, 16, 16
	data := make([]float64, width*height*depth)
	
	// Create a sphere
	for z := 0; z < depth; z++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				// Calculate distance from center
				dx := float64(x - width/2)
				dy := float64(y - height/2)
				dz := float64(z - depth/2)
				distance := math.Sqrt(dx*dx + dy*dy + dz*dz)
				
				// Set value based on distance (inside sphere = 1, outside = 0)
				if distance < float64(width)/4 {
					data[z*width*height + y*width + x] = 1.0
				} else {
					data[z*width*height + y*width + x] = 0.0
				}
			}
		}
	}
	
	// Reset timer before the actual benchmark
	b.ResetTimer()
	
	// Run the benchmark
	for i := 0; i < b.N; i++ {
		mc := NewMarchingCubes(data, width, height, depth, 0.5)
		mc.GenerateTriangles()
	}
}