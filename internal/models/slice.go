package models

import (
	"image"
)

// Slice represents a single MRI slice with metadata
type Slice struct {
	// Image is the actual slice image data
	Image image.Image
	
	// Index is the position of this slice in the sequence
	Index int
	
	// Filename is the original filename of the slice
	Filename string
	
	// Thickness is the physical thickness of the slice in mm
	Thickness float64
	
	// Position is the physical position of the slice along the axis
	Position float64
}

// Volume represents a 3D volume reconstructed from MRI slices
type Volume struct {
	// Data is the 3D volume data as a 1D array in row-major order
	Data []float64
	
	// Width is the width of the volume in voxels
	Width int
	
	// Height is the height of the volume in voxels
	Height int
	
	// Depth is the depth of the volume in voxels
	Depth int
	
	// VoxelSize is the physical size of each voxel in mm
	VoxelSize struct {
		X, Y, Z float64
	}
}

// Quadrant represents one of the four quadrants of a slice
// as described in the paper for parallel processing
type Quadrant int

const (
	TopLeft Quadrant = iota
	TopRight
	BottomLeft
	BottomRight
)

// SubVolume represents a portion of the volume for parallel processing
// as described in Algorithm 1 in the paper
type SubVolume struct {
	// Data is the 3D sub-volume data as a 1D array
	Data []float64
	
	// Width, Height, Depth are the dimensions of the sub-volume
	Width, Height, Depth int
	
	// QuadrantType indicates which quadrant this sub-volume belongs to
	QuadrantType Quadrant
	
	// SubsetIndex indicates which subset this sub-volume belongs to
	// (the paper divides each quadrant into 2 subsets)
	SubsetIndex int
} 