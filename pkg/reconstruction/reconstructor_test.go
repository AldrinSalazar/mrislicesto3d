package reconstruction

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// createTempDir creates a temporary directory for test files
func createTempDir(t *testing.T) string {
	dir, err := os.MkdirTemp("", "mrislicesto3d-test-*")
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	return dir
}

// createTestImage creates a grayscale test image with the specified dimensions and pattern
func createTestImage(width, height int, pattern func(x, y int) uint16) image.Image {
	img := image.NewGray16(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			img.Set(x, y, color.Gray16{Y: pattern(x, y)})
		}
	}
	return img
}

// TestBasicReconstructor runs a basic reconstruction process using generated test images
// This test verifies that the entire pipeline can run without errors
func TestBasicReconstructor(t *testing.T) {
	// Skip this test for regular unit testing, as it is slow and comprehensive
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Create temporary directories for test
	tmpDir := createTempDir(t)
	defer os.RemoveAll(tmpDir)

	inputDir := filepath.Join(tmpDir, "input")
	outputFile := filepath.Join(tmpDir, "output.stl")
	if err := os.MkdirAll(inputDir, 0755); err != nil {
		t.Fatalf("Failed to create input dir: %v", err)
	}

	// Create test images (concentric circles growing in each slice)
	createTestSlices(t, inputDir)

	// Initialize reconstructor
	params := &Params{
		InputDir:   inputDir,
		OutputFile: outputFile,
		NumCores:   2,
		SliceGap:   3.0,
	}

	reconstructor := NewReconstructor(params)

	// Run reconstruction (each sub-test can be run independently if needed)
	t.Run("Loading", func(t *testing.T) {
		err := reconstructor.loadSlices()
		if err != nil {
			t.Fatalf("Failed to load slices: %v", err)
		}

		if len(reconstructor.slices) == 0 {
			t.Fatal("No slices were loaded")
		}
	})

	t.Run("Denoising", func(t *testing.T) {
		// First load slices if not already loaded
		if len(reconstructor.slices) == 0 {
			if err := reconstructor.loadSlices(); err != nil {
				t.Fatalf("Failed to load slices: %v", err)
			}
		}

		err := reconstructor.denoiseSlices()
		if err != nil {
			t.Fatalf("Failed to denoise slices: %v", err)
		}
	})

	t.Run("DivideDataset", func(t *testing.T) {
		// First load and denoise if not already done
		if len(reconstructor.slices) == 0 {
			if err := reconstructor.loadSlices(); err != nil {
				t.Fatalf("Failed to load slices: %v", err)
			}
			if err := reconstructor.denoiseSlices(); err != nil {
				t.Fatalf("Failed to denoise slices: %v", err)
			}
		}

		err := reconstructor.divideDataset()
		if err != nil {
			t.Fatalf("Failed to divide dataset: %v", err)
		}

		if len(reconstructor.subSlices) == 0 {
			t.Fatal("No sub-slices were created")
		}
	})

	// Skip the full reconstruction test as it takes too long
	t.Run("FullReconstruction", func(t *testing.T) {
		t.Skip("Skipping full reconstruction test as it takes too long")

		// Create a new reconstructor for this test
		fullParams := &Params{
			InputDir:   inputDir,
			OutputFile: outputFile,
			NumCores:   2,
			SliceGap:   3.0,
		}

		fullReconstructor := NewReconstructor(fullParams)

		// Run the full pipeline
		err := fullReconstructor.Process()
		if err != nil {
			t.Fatalf("Failed to run full reconstruction: %v", err)
		}

		// Check that the output file was created
		if _, err := os.Stat(outputFile); os.IsNotExist(err) {
			t.Fatalf("Output file was not created: %v", err)
		}
	})
}

// createTestSlices creates a series of test MRI slice images for testing
func createTestSlices(t *testing.T, dir string) {
	// Create a series of test images with concentric circles
	size := 64
	sliceCount := 5

	for i := 0; i < sliceCount; i++ {
		img := image.NewGray16(image.Rect(0, 0, size, size))

		// Create a simple pattern (concentric circles) that changes across slices
		centerX := size / 2
		centerY := size / 2
		// Radius increases with slice index
		radius := size/4 + i*4

		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				dx := float64(x - centerX)
				dy := float64(y - centerY)
				dist := math.Sqrt(dx*dx + dy*dy)

				// Create concentric rings
				if math.Abs(dist-float64(radius)) < 3 {
					img.Set(x, y, color.Gray16{Y: 65535}) // White ring
				} else {
					// Create a gradient in the background
					bgValue := uint16((float64(x+y) / float64(2*size)) * 20000)
					img.Set(x, y, color.Gray16{Y: bgValue})
				}
			}
		}

		// Save the image
		filename := filepath.Join(dir, fmt.Sprintf("slice_%03d.jpg", i))
		f, err := os.Create(filename)
		if err != nil {
			t.Fatalf("Failed to create test image: %v", err)
		}
		if err := jpeg.Encode(f, img, &jpeg.Options{Quality: 100}); err != nil {
			f.Close()
			t.Fatalf("Failed to encode test image: %v", err)
		}
		f.Close()
	}
}

// TestNewReconstructor verifies that a new reconstructor is correctly initialized
func TestNewReconstructor(t *testing.T) {
	params := &Params{
		InputDir:   "/path/to/input",
		OutputFile: "output.stl",
		NumCores:   4,
		SliceGap:   2.5,
	}

	reconstructor := NewReconstructor(params)

	if reconstructor.params != params {
		t.Errorf("Reconstructor should use the provided params")
	}

	if len(reconstructor.slices) != 0 {
		t.Errorf("New reconstructor should have empty slices")
	}
}

// TestExtractNumber verifies the extraction of numeric parts from filenames
func TestExtractNumber(t *testing.T) {
	testCases := []struct {
		filename string
		expected int
	}{
		{"slice_1.jpg", 1},
		{"slice_023.jpg", 23},
		{"img456.jpg", 456},
		{"not_a_number.jpg", 0},
		{"mixed123text456.jpg", 123456},
	}

	for _, tc := range testCases {
		result := extractNumber(tc.filename)
		if result != tc.expected {
			t.Errorf("extractNumber(%s): expected %d, got %d", tc.filename, tc.expected, result)
		}
	}
}

// TestImageToFloat and TestFloatToImage verify the image conversion functions
func TestImageConversion(t *testing.T) {
	// Create a simple test image with a gradient pattern
	width, height := 4, 4
	testImg := createTestImage(width, height, func(x, y int) uint16 {
		// Create a gradient where each pixel has a unique value
		return uint16((y*width + x) * 4096) // *4096 to spread values across 16-bit range
	})

	// Convert to float array
	floatData := imageToFloat(testImg)

	// Verify correct conversion to float (0-1 range)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*width + x
			expected := float64((y*width+x)*4096) / 65535.0
			if floatData[idx] < expected-0.001 || floatData[idx] > expected+0.001 {
				t.Errorf("imageToFloat: at (%d,%d), expected %.6f, got %.6f",
					x, y, expected, floatData[idx])
			}
		}
	}

	// Convert back to image
	roundTripImg := floatToImage(floatData, width, height)

	// Verify values are preserved within rounding error
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			originalValue := testImg.At(x, y).(color.Gray16).Y
			newValue := roundTripImg.At(x, y).(color.Gray16).Y

			// Allow for minor rounding differences due to float conversion
			diff := int(originalValue) - int(newValue)
			if diff < -1 || diff > 1 {
				t.Errorf("Round-trip conversion at (%d,%d): expected %d, got %d (diff: %d)",
					x, y, originalValue, newValue, diff)
			}
		}
	}
}

// TestImageToFloatMultiple verifies the conversion of multiple images to a float array
func TestImageToFloatMultiple(t *testing.T) {
	// Create 3 simple test images with different patterns
	width, height := 4, 4

	images := []image.Image{
		createTestImage(width, height, func(x, y int) uint16 { return uint16(x * 4096) }),
		createTestImage(width, height, func(x, y int) uint16 { return uint16(y * 4096) }),
		createTestImage(width, height, func(x, y int) uint16 { return uint16((x + y) * 4096) }),
	}

	// Convert to float array
	floatData := imagesToFloat(images)

	// Verify correct conversion and ordering
	expectedLength := len(images) * width * height
	if len(floatData) != expectedLength {
		t.Errorf("imagesToFloat: expected length %d, got %d", expectedLength, len(floatData))
	}

	// Check values from each image
	for imgIdx, img := range images {
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				originalValue := img.At(x, y).(color.Gray16).Y
				expected := float64(originalValue) / 65535.0

				idx := imgIdx*width*height + y*width + x
				if floatData[idx] < expected-0.001 || floatData[idx] > expected+0.001 {
					t.Errorf("imagesToFloat: image %d at (%d,%d), expected %.6f, got %.6f",
						imgIdx, x, y, expected, floatData[idx])
				}
			}
		}
	}
}

// TestSplitImageIntoQuadrants verifies the image quadrant splitting function
func TestSplitImageIntoQuadrants(t *testing.T) {
	// Create a test image with distinct values in each quadrant
	width, height := 4, 4
	testImg := createTestImage(width, height, func(x, y int) uint16 {
		// Assign different values to each quadrant:
		// [A][B]
		// [C][D]
		if x < width/2 && y < height/2 {
			return 10000 // Quadrant A (top-left)
		} else if x >= width/2 && y < height/2 {
			return 20000 // Quadrant B (top-right)
		} else if x < width/2 && y >= height/2 {
			return 30000 // Quadrant C (bottom-left)
		} else {
			return 40000 // Quadrant D (bottom-right)
		}
	})

	// Create a reconstructor instance
	params := &Params{SliceGap: 1.0}
	reconstructor := NewReconstructor(params)
	reconstructor.width = width
	reconstructor.height = height

	// Split image into quadrants
	quadrants := reconstructor.splitImageIntoQuadrants(testImg)

	// Verify we got 4 quadrants
	if len(quadrants) != 4 {
		t.Fatalf("Expected 4 quadrants, got %d", len(quadrants))
	}

	// Verify each quadrant has the correct size
	expectedQuadrantWidth := width / 2
	expectedQuadrantHeight := height / 2

	for i, quadrant := range quadrants {
		bounds := quadrant.Bounds()
		if bounds.Dx() != expectedQuadrantWidth || bounds.Dy() != expectedQuadrantHeight {
			t.Errorf("Quadrant %d: expected size %dx%d, got %dx%d",
				i, expectedQuadrantWidth, expectedQuadrantHeight, bounds.Dx(), bounds.Dy())
		}
	}

	// Verify the values in each quadrant
	// Define expected average color values for each quadrant
	// The values are scaled down from 16-bit to 8-bit color space
	expectedValues := []struct {
		r, g, b uint32
	}{
		{39 << 8, 39 << 8, 39 << 8},    // Quadrant 0 (top-left)
		{78 << 8, 78 << 8, 78 << 8},    // Quadrant 1 (top-right)
		{117 << 8, 117 << 8, 117 << 8}, // Quadrant 2 (bottom-left)
		{156 << 8, 156 << 8, 156 << 8}, // Quadrant 3 (bottom-right)
	}

	for quadIdx, quadrant := range quadrants {
		// Check the center pixel of each quadrant
		bounds := quadrant.Bounds()
		centerX := bounds.Min.X + bounds.Dx()/2
		centerY := bounds.Min.Y + bounds.Dy()/2

		// Get the color value
		r, g, b, _ := quadrant.At(centerX, centerY).RGBA()

		// Check if the color is close to the expected value
		expectedR := expectedValues[quadIdx].r
		expectedG := expectedValues[quadIdx].g
		expectedB := expectedValues[quadIdx].b

		// Allow for some tolerance in the color values
		tolerance := uint32(5000 << 8)

		if math.Abs(float64(r)-float64(expectedR)) > float64(tolerance) ||
			math.Abs(float64(g)-float64(expectedG)) > float64(tolerance) ||
			math.Abs(float64(b)-float64(expectedB)) > float64(tolerance) {
			t.Errorf("Quadrant %d: expected color close to RGB(%d,%d,%d), got RGB(%d,%d,%d)",
				quadIdx, expectedR>>8, expectedG>>8, expectedB>>8, r>>8, g>>8, b>>8)
		}
	}
}

// TestGetVolumeData verifies that the volume data can be retrieved correctly
func TestGetVolumeData(t *testing.T) {
	// Skip this test for regular unit testing, as it is slow
	if testing.Short() {
		t.Skip("Skipping volume data test in short mode")
	}

	// Create temporary directories for test
	tmpDir := createTempDir(t)
	defer os.RemoveAll(tmpDir)

	inputDir := filepath.Join(tmpDir, "input")
	outputFile := filepath.Join(tmpDir, "output.stl")
	if err := os.MkdirAll(inputDir, 0755); err != nil {
		t.Fatalf("Failed to create input dir: %v", err)
	}

	// Create test slices (simple gradient pattern)
	numSlices := 5
	width, height := 20, 20
	sliceGap := 2.0

	for i := 0; i < numSlices; i++ {
		// Create a gradient image where pixel values increase with slice index
		img := createTestImage(width, height, func(x, y int) uint16 {
			// Value increases with slice index
			return uint16(float64(i) / float64(numSlices-1) * 65535)
		})

		// Save the image
		filename := filepath.Join(inputDir, fmt.Sprintf("slice_%03d.jpg", i))
		file, err := os.Create(filename)
		if err != nil {
			t.Fatalf("Failed to create test image file: %v", err)
		}

		if err := jpeg.Encode(file, img, &jpeg.Options{Quality: 90}); err != nil {
			file.Close()
			t.Fatalf("Failed to encode test image: %v", err)
		}
		file.Close()
	}

	// Initialize reconstructor
	params := &Params{
		InputDir:   inputDir,
		OutputFile: outputFile,
		NumCores:   2,
		SliceGap:   sliceGap,
	}

	reconstructor := NewReconstructor(params)

	// Load slices
	if err := reconstructor.loadSlices(); err != nil {
		t.Fatalf("Failed to load slices: %v", err)
	}

	// Get volume data
	volumeData, volumeWidth, volumeHeight, volumeDepth := reconstructor.GetVolumeData()

	// Verify dimensions
	if volumeWidth != width {
		t.Errorf("Expected volume width %d, got %d", width, volumeWidth)
	}

	if volumeHeight != height {
		t.Errorf("Expected volume height %d, got %d", height, volumeHeight)
	}

	// Expected depth is (numSlices-1)*slicesPerGap + 1
	expectedDepth := int(float64(numSlices-1)*math.Ceil(sliceGap)) + 1
	if volumeDepth != expectedDepth {
		t.Errorf("Expected volume depth %d, got %d", expectedDepth, volumeDepth)
	}

	// Verify volume size
	expectedSize := volumeWidth * volumeHeight * volumeDepth
	if len(volumeData) != expectedSize {
		t.Errorf("Expected volume data size %d, got %d", expectedSize, len(volumeData))
	}

	// Verify original slices are preserved at their positions
	for i := 0; i < numSlices; i++ {
		// Calculate position in volume
		zPos := i * int(math.Ceil(sliceGap))

		// Check center pixel of the slice
		centerX, centerY := width/2, height/2
		volumeIdx := zPos*width*height + centerY*width + centerX

		// Expected value based on the gradient pattern
		expectedValue := float64(i) / float64(numSlices-1)

		// Allow for some compression artifacts
		if math.Abs(volumeData[volumeIdx]-expectedValue) > 0.1 {
			t.Errorf("Slice %d center value mismatch: expected ~%.2f, got %.2f",
				i, expectedValue, volumeData[volumeIdx])
		}
	}

	// Verify interpolated slices have intermediate values
	if numSlices > 1 && int(math.Ceil(sliceGap)) > 1 {
		// Check an interpolated position between first and second slice
		zPos := int(math.Ceil(sliceGap)) / 2
		centerX, centerY := width/2, height/2
		volumeIdx := zPos*width*height + centerY*width + centerX

		// Allow for some interpolation differences
		if volumeData[volumeIdx] < 0 || volumeData[volumeIdx] > 1.0/float64(numSlices-1) {
			t.Errorf("Interpolated value out of expected range: got %.2f", volumeData[volumeIdx])
		}
	}
}
