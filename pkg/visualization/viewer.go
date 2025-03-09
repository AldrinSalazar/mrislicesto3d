package visualization

import (
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"
	"path/filepath"
)

// Viewer implements the 3D visualization functionality described in Algorithm 3
// of the paper "Fast 3D Volumetric Image Reconstruction from 2D MRI Slices by
// Parallel Processing" by Somoballi Ghoshal et al.
type Viewer struct {
	// volumeData holds the 3D reconstructed volume data
	volumeData []float64
	
	// dimensions of the volume
	width  int
	height int
	depth  int
	
	// sliceGap is the physical distance between consecutive slices in mm
	sliceGap float64
}

// NewViewer creates a new 3D visualization viewer
func NewViewer(volumeData []float64, width, height, depth int, sliceGap float64) *Viewer {
	return &Viewer{
		volumeData: volumeData,
		width:      width,
		height:     height,
		depth:      depth,
		sliceGap:   sliceGap,
	}
}

// ExtractSlice extracts a 2D slice from the 3D volume along the specified axis
// This implements part of Algorithm 3 from the paper
func (v *Viewer) ExtractSlice(axis string, position int) (image.Image, error) {
	if position < 0 {
		return nil, fmt.Errorf("position must be non-negative")
	}
	
	var img image.Gray16
	
	switch axis {
	case "x", "X":
		// Extract slice along YZ plane
		if position >= v.width {
			return nil, fmt.Errorf("position %d exceeds width %d", position, v.width)
		}
		
		img = *image.NewGray16(image.Rect(0, 0, v.depth, v.height))
		for y := 0; y < v.height; y++ {
			for z := 0; z < v.depth; z++ {
				idx := z*v.width*v.height + y*v.width + position
				if idx < len(v.volumeData) {
					value := uint16(math.Max(0, math.Min(65535, v.volumeData[idx]*65535)))
					img.SetGray16(z, y, color.Gray16{Y: value})
				}
			}
		}
		
	case "y", "Y":
		// Extract slice along XZ plane
		if position >= v.height {
			return nil, fmt.Errorf("position %d exceeds height %d", position, v.height)
		}
		
		img = *image.NewGray16(image.Rect(0, 0, v.width, v.depth))
		for z := 0; z < v.depth; z++ {
			for x := 0; x < v.width; x++ {
				idx := z*v.width*v.height + position*v.width + x
				if idx < len(v.volumeData) {
					value := uint16(math.Max(0, math.Min(65535, v.volumeData[idx]*65535)))
					img.SetGray16(x, z, color.Gray16{Y: value})
				}
			}
		}
		
	case "z", "Z":
		// Extract slice along XY plane
		if position >= v.depth {
			return nil, fmt.Errorf("position %d exceeds depth %d", position, v.depth)
		}
		
		img = *image.NewGray16(image.Rect(0, 0, v.width, v.height))
		for y := 0; y < v.height; y++ {
			for x := 0; x < v.width; x++ {
				idx := position*v.width*v.height + y*v.width + x
				if idx < len(v.volumeData) {
					value := uint16(math.Max(0, math.Min(65535, v.volumeData[idx]*65535)))
					img.SetGray16(x, y, color.Gray16{Y: value})
				}
			}
		}
		
	default:
		return nil, fmt.Errorf("invalid axis: %s (must be x, y, or z)", axis)
	}
	
	return &img, nil
}

// ExtractRegion extracts a 3D subregion from the volume
// This implements the region extraction part of Algorithm 3 from the paper
func (v *Viewer) ExtractRegion(startX, startY, startZ, sizeX, sizeY, sizeZ int) ([]float64, error) {
	// Validate parameters
	if startX < 0 || startY < 0 || startZ < 0 {
		return nil, fmt.Errorf("start coordinates must be non-negative")
	}
	
	if sizeX <= 0 || sizeY <= 0 || sizeZ <= 0 {
		return nil, fmt.Errorf("size dimensions must be positive")
	}
	
	if startX+sizeX > v.width || startY+sizeY > v.height || startZ+sizeZ > v.depth {
		return nil, fmt.Errorf("region extends beyond volume boundaries")
	}
	
	// Create output region
	region := make([]float64, sizeX*sizeY*sizeZ)
	
	// Extract the region
	for z := 0; z < sizeZ; z++ {
		for y := 0; y < sizeY; y++ {
			for x := 0; x < sizeX; x++ {
				// Calculate indices
				srcIdx := (startZ+z)*v.width*v.height + (startY+y)*v.width + (startX+x)
				dstIdx := z*sizeX*sizeY + y*sizeX + x
				
				// Copy data
				if srcIdx < len(v.volumeData) && dstIdx < len(region) {
					region[dstIdx] = v.volumeData[srcIdx]
				}
			}
		}
	}
	
	return region, nil
}

// SaveSlice saves an extracted slice as a JPEG image
func (v *Viewer) SaveSlice(img image.Image, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	return jpeg.Encode(file, img, &jpeg.Options{Quality: 90})
}

// SaveSliceSequence extracts and saves a sequence of slices along the specified axis
func (v *Viewer) SaveSliceSequence(axis string, outputDir string) error {
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return err
	}
	
	var maxPos int
	switch axis {
	case "x", "X":
		maxPos = v.width
	case "y", "Y":
		maxPos = v.height
	case "z", "Z":
		maxPos = v.depth
	default:
		return fmt.Errorf("invalid axis: %s (must be x, y, or z)", axis)
	}
	
	for pos := 0; pos < maxPos; pos++ {
		img, err := v.ExtractSlice(axis, pos)
		if err != nil {
			return err
		}
		
		filename := filepath.Join(outputDir, fmt.Sprintf("slice_%s_%03d.jpg", axis, pos))
		if err := v.SaveSlice(img, filename); err != nil {
			return err
		}
	}
	
	return nil
} 