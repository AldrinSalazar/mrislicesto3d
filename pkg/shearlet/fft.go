package shearlet

import (
	"math"

	"gonum.org/v1/gonum/dsp/fourier"
)

// fft2D performs a 2D Fast Fourier Transform on the input data.
// This is a key component of the shearlet transform as it allows efficient
// filtering in the frequency domain.
//
// Parameters:
//   - data: Input image data as a 1D array (row-major order)
//   - size: Width/height of the square image
//
// Returns:
//   - The 2D FFT of the input data as a 1D array of complex numbers
func (t *Transform) fft2D(data []float64, size int) []complex128 {
	// Create a new FFT object from Gonum
	fft := fourier.NewFFT(size)
	
	// Allocate memory for the result
	result := make([]complex128, size*size)
	
	// Temporary storage for row and column FFTs
	rowInput := make([]float64, size)
	rowOutput := make([]complex128, size/2+1) // Gonum FFT output size for real input
	
	// Perform row-wise FFT
	for i := 0; i < size; i++ {
		// Extract row
		for j := 0; j < size; j++ {
			rowInput[j] = data[i*size+j]
		}
		
		// Compute FFT of the row
		fft.Coefficients(rowOutput, rowInput)
		
		// Convert to full complex FFT (since we need the full spectrum)
		fullRowOutput := make([]complex128, size)
		for j := 0; j < len(rowOutput); j++ {
			fullRowOutput[j] = rowOutput[j]
		}
		for j := len(rowOutput); j < size; j++ {
			// Use conjugate symmetry: F(n-k) = F*(k)
			k := size - j
			if k < len(rowOutput) {
				fullRowOutput[j] = complex(real(rowOutput[k]), -imag(rowOutput[k]))
			}
		}
		
		// Store row FFT results
		for j := 0; j < size; j++ {
			result[i*size+j] = fullRowOutput[j]
		}
	}
	
	// Temporary storage for column FFT
	colInput := make([]complex128, size)
	
	// For column-wise FFT, we need to use a different approach since the input is complex
	// We'll use our own implementation for complex FFT
	for j := 0; j < size; j++ {
		// Extract column from row FFT results
		for i := 0; i < size; i++ {
			colInput[i] = result[i*size+j]
		}
		
		// Compute FFT of the column using our own implementation
		colOutput := complexFFT(colInput)
		
		// Store column FFT results
		for i := 0; i < size; i++ {
			result[i*size+j] = colOutput[i]
		}
	}
	
	return result
}

// complexFFT performs a 1D FFT on complex input data
// This is a recursive implementation of the Cooley-Tukey algorithm
func complexFFT(x []complex128) []complex128 {
	n := len(x)
	if n <= 1 {
		return x
	}

	// Split into even and odd
	even := make([]complex128, n/2)
	odd := make([]complex128, n/2)
	for i := 0; i < n/2; i++ {
		even[i] = x[2*i]
		odd[i] = x[2*i+1]
	}

	// Recursive FFT
	even = complexFFT(even)
	odd = complexFFT(odd)

	// Combine results
	result := make([]complex128, n)
	for k := 0; k < n/2; k++ {
		t := complex(
			math.Cos(-2*math.Pi*float64(k)/float64(n)),
			math.Sin(-2*math.Pi*float64(k)/float64(n)),
		) * odd[k]
		result[k] = even[k] + t
		result[k+n/2] = even[k] - t
	}

	return result
} 