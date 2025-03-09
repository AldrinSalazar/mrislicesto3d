package visualization

import (
	"fmt"
	"image"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestNewViewer verifies that a new viewer is created with the correct parameters
func TestNewViewer(t *testing.T) {
	// Create test data
	width, height, depth := 10, 10, 5
	volumeData := make([]float64, width*height*depth)
	sliceGap := 2.0
	
	// Fill with test pattern
	for z := 0; z < depth; z++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				idx := z*width*height + y*width + x
				volumeData[idx] = float64(x+y+z) / float64(width+height+depth)
			}
		}
	}
	
	// Create viewer
	viewer := NewViewer(volumeData, width, height, depth, sliceGap)
	
	// Verify parameters
	if viewer.width != width {
		t.Errorf("Expected width %d, got %d", width, viewer.width)
	}
	
	if viewer.height != height {
		t.Errorf("Expected height %d, got %d", height, viewer.height)
	}
	
	if viewer.depth != depth {
		t.Errorf("Expected depth %d, got %d", depth, viewer.depth)
	}
	
	if viewer.sliceGap != sliceGap {
		t.Errorf("Expected slice gap %f, got %f", sliceGap, viewer.sliceGap)
	}
	
	if len(viewer.volumeData) != len(volumeData) {
		t.Errorf("Expected volume data length %d, got %d", len(volumeData), len(viewer.volumeData))
	}
}

// TestExtractSlice verifies that slices are correctly extracted from the volume
func TestExtractSlice(t *testing.T) {
	// Create test data
	width, height, depth := 10, 10, 5
	volumeData := make([]float64, width*height*depth)
	
	// Fill with test pattern: each slice along Z has a unique value
	for z := 0; z < depth; z++ {
		value := float64(z) / float64(depth)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				idx := z*width*height + y*width + x
				volumeData[idx] = value
			}
		}
	}
	
	// Create viewer
	viewer := NewViewer(volumeData, width, height, depth, 1.0)
	
	// Test extracting Z slices
	for z := 0; z < depth; z++ {
		img, err := viewer.ExtractSlice("z", z)
		if err != nil {
			t.Fatalf("Failed to extract Z slice at position %d: %v", z, err)
		}
		
		// Verify dimensions
		bounds := img.Bounds()
		if bounds.Dx() != width || bounds.Dy() != height {
			t.Errorf("Expected Z slice dimensions %dx%d, got %dx%d", 
				width, height, bounds.Dx(), bounds.Dy())
		}
		
		// Verify pixel values (sample a few points)
		expectedValue := uint16(math.Max(0, math.Min(65535, float64(z)/float64(depth)*65535)))
		gray16Img, ok := img.(*image.Gray16)
		if !ok {
			t.Fatalf("Expected *image.Gray16, got %T", img)
		}
		
		// Check center pixel
		centerX, centerY := width/2, height/2
		centerValue := gray16Img.Gray16At(centerX, centerY).Y
		if math.Abs(float64(centerValue-expectedValue)) > 1.0 {
			t.Errorf("Expected Z slice value ~%d at center, got %d", 
				expectedValue, centerValue)
		}
	}
	
	// Test extracting X slice
	xPos := width / 2
	imgX, err := viewer.ExtractSlice("x", xPos)
	if err != nil {
		t.Fatalf("Failed to extract X slice: %v", err)
	}
	
	// Verify dimensions
	boundsX := imgX.Bounds()
	if boundsX.Dx() != depth || boundsX.Dy() != height {
		t.Errorf("Expected X slice dimensions %dx%d, got %dx%d", 
			depth, height, boundsX.Dx(), boundsX.Dy())
	}
	
	// Test extracting Y slice
	yPos := height / 2
	imgY, err := viewer.ExtractSlice("y", yPos)
	if err != nil {
		t.Fatalf("Failed to extract Y slice: %v", err)
	}
	
	// Verify dimensions
	boundsY := imgY.Bounds()
	if boundsY.Dx() != width || boundsY.Dy() != depth {
		t.Errorf("Expected Y slice dimensions %dx%d, got %dx%d", 
			width, depth, boundsY.Dx(), boundsY.Dy())
	}
	
	// Test invalid axis
	_, err = viewer.ExtractSlice("invalid", 0)
	if err == nil {
		t.Error("Expected error for invalid axis, got nil")
	}
	
	// Test out of bounds position
	_, err = viewer.ExtractSlice("z", depth+1)
	if err == nil {
		t.Error("Expected error for out of bounds position, got nil")
	}
}

// TestExtractRegion verifies that 3D regions are correctly extracted
func TestExtractRegion(t *testing.T) {
	// Create test data
	width, height, depth := 10, 10, 5
	volumeData := make([]float64, width*height*depth)
	
	// Fill with test pattern: gradient along each axis
	for z := 0; z < depth; z++ {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				idx := z*width*height + y*width + x
				volumeData[idx] = float64(x)/float64(width) + 
					float64(y)/float64(height) + 
					float64(z)/float64(depth)
			}
		}
	}
	
	// Create viewer
	viewer := NewViewer(volumeData, width, height, depth, 1.0)
	
	// Extract a region
	startX, startY, startZ := 2, 3, 1
	sizeX, sizeY, sizeZ := 4, 3, 2
	
	region, err := viewer.ExtractRegion(startX, startY, startZ, sizeX, sizeY, sizeZ)
	if err != nil {
		t.Fatalf("Failed to extract region: %v", err)
	}
	
	// Verify region size
	expectedSize := sizeX * sizeY * sizeZ
	if len(region) != expectedSize {
		t.Errorf("Expected region size %d, got %d", expectedSize, len(region))
	}
	
	// Verify region values (sample a few points)
	for z := 0; z < sizeZ; z++ {
		for y := 0; y < sizeY; y++ {
			for x := 0; x < sizeX; x++ {
				// Calculate indices
				regionIdx := z*sizeX*sizeY + y*sizeX + x
				volumeIdx := (startZ+z)*width*height + (startY+y)*width + (startX+x)
				
				// Compare values
				if region[regionIdx] != volumeData[volumeIdx] {
					t.Errorf("Region value mismatch at (%d,%d,%d): expected %f, got %f",
						x, y, z, volumeData[volumeIdx], region[regionIdx])
				}
			}
		}
	}
	
	// Test invalid parameters
	_, err = viewer.ExtractRegion(-1, 0, 0, 1, 1, 1)
	if err == nil {
		t.Error("Expected error for negative start coordinate, got nil")
	}
	
	_, err = viewer.ExtractRegion(0, 0, 0, 0, 1, 1)
	if err == nil {
		t.Error("Expected error for zero size, got nil")
	}
	
	_, err = viewer.ExtractRegion(width-1, 0, 0, 2, 1, 1)
	if err == nil {
		t.Error("Expected error for region extending beyond volume, got nil")
	}
}

// TestSaveSlice verifies that slices can be saved to disk
func TestSaveSlice(t *testing.T) {
	// Skip this test in short mode
	if testing.Short() {
		t.Skip("Skipping file I/O test in short mode")
	}
	
	// Create temporary directory
	tempDir, err := os.MkdirTemp("", "viewer-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	// Create test data
	width, height, depth := 10, 10, 5
	volumeData := make([]float64, width*height*depth)
	for i := range volumeData {
		volumeData[i] = 0.5 // Mid-gray
	}
	
	// Create viewer
	viewer := NewViewer(volumeData, width, height, depth, 1.0)
	
	// Extract a slice
	img, err := viewer.ExtractSlice("z", 0)
	if err != nil {
		t.Fatalf("Failed to extract slice: %v", err)
	}
	
	// Save the slice
	filename := filepath.Join(tempDir, "test_slice.jpg")
	err = viewer.SaveSlice(img, filename)
	if err != nil {
		t.Fatalf("Failed to save slice: %v", err)
	}
	
	// Verify file exists
	if _, err := os.Stat(filename); os.IsNotExist(err) {
		t.Errorf("Saved file does not exist: %s", filename)
	}
}

// TestSaveSliceSequence verifies that a sequence of slices can be saved
func TestSaveSliceSequence(t *testing.T) {
	// Skip this test in short mode
	if testing.Short() {
		t.Skip("Skipping file I/O test in short mode")
	}
	
	// Create temporary directory
	tempDir, err := os.MkdirTemp("", "viewer-sequence-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)
	
	// Create test data
	width, height, depth := 5, 5, 3
	volumeData := make([]float64, width*height*depth)
	for i := range volumeData {
		volumeData[i] = 0.5 // Mid-gray
	}
	
	// Create viewer
	viewer := NewViewer(volumeData, width, height, depth, 1.0)
	
	// Save slice sequence
	outputDir := filepath.Join(tempDir, "slices")
	err = viewer.SaveSliceSequence("z", outputDir)
	if err != nil {
		t.Fatalf("Failed to save slice sequence: %v", err)
	}
	
	// Verify files exist
	for z := 0; z < depth; z++ {
		filename := filepath.Join(outputDir, fmt.Sprintf("slice_z_%03d.jpg", z))
		if _, err := os.Stat(filename); os.IsNotExist(err) {
			t.Errorf("Expected slice file does not exist: %s", filename)
		}
	}
	
	// Test invalid axis
	err = viewer.SaveSliceSequence("invalid", outputDir)
	if err == nil {
		t.Error("Expected error for invalid axis, got nil")
	}
} 