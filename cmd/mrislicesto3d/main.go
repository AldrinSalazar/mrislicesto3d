package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"mrislicesto3d/pkg/reconstruction"
	"mrislicesto3d/pkg/visualization"
)

func main() {
	// Parse command line arguments
	inputDir := flag.String("input", "", "Directory containing 2D MRI slices")
	outputName := flag.String("output", "output.stl", "Output STL filename")
	numCores := flag.Int("cores", runtime.NumCPU(), "Number of CPU cores to use (default: all available)")
	sliceGap := flag.Float64("gap", 1.5, "Inter-slice gap in mm")
	extractSlices := flag.Bool("extract-slices", false, "Extract and save reconstructed slices along all axes")
	slicesDir := flag.String("slices-dir", "reconstructed_slices", "Directory to save extracted slices")
	saveIntermediary := flag.Bool("save-intermediary", true, "Save intermediary results during processing")
	intermediaryDir := flag.String("intermediary-dir", "intermediary_results", "Directory to save intermediary results")
	flag.Parse()

	// Validate inputs
	if *inputDir == "" {
		flag.Usage()
		os.Exit(1)
	}

	// Get executable directory for output
	execPath, err := os.Executable()
	if err != nil {
		log.Fatalf("Failed to get executable path: %v", err)
	}
	outputDir := filepath.Dir(execPath)
	outputPath := filepath.Join(outputDir, *outputName)
	
	// Create intermediary directory path
	intermediaryPath := filepath.Join(outputDir, *intermediaryDir)

	fmt.Println("================================")
	fmt.Println("FAST 3D VOLUMETRIC IMAGE RECONSTRUCTION FROM 2D MRI SLICES BY PARALLEL PROCESSING")
	fmt.Println("Based on the paper by Somoballi Ghoshal et al.")
	fmt.Println("================================")

	// Initialize reconstruction parameters
	params := &reconstruction.Params{
		InputDir:               *inputDir,
		OutputFile:             outputPath,
		NumCores:               *numCores,
		SliceGap:               *sliceGap,
		SaveIntermediaryResults: *saveIntermediary,
		IntermediaryDir:        intermediaryPath,
	}

	// Create reconstructor instance
	reconstructor := reconstruction.NewReconstructor(params)

	// Run the reconstruction pipeline
	fmt.Println("Starting 3D reconstruction with parallel processing...")
	startTime := time.Now()
	if err := reconstructor.Process(); err != nil {
		log.Fatalf("Reconstruction failed: %v", err)
	}
	processingTime := time.Since(startTime)

	// Get and display validation metrics as shown in the paper
	metrics := reconstructor.GetMetrics()
	fmt.Printf("\nReconstruction completed successfully in %.2f seconds!\n", processingTime.Seconds())
	fmt.Printf("Output 3D model saved to: %s\n\n", outputPath)
	
	fmt.Printf("Validation Metrics (Table 2 from paper):\n")
	fmt.Printf("=======================================\n")
	fmt.Printf("Mutual Information (MI): %.3f\n", metrics.MI)
	fmt.Printf("Entropy Difference: %.3f\n", metrics.EntropyDiff)
	fmt.Printf("Root Mean Square Error (RMSE): %.6f\n", metrics.RMSE)
	fmt.Printf("Structural Similarity Index (SSIM): %.3f\n", metrics.SSIM)
	fmt.Printf("Edge Preservation Ratio: %.3f\n", metrics.EdgePreserved)
	fmt.Printf("Overall Accuracy: %.2f%%\n", metrics.Accuracy)
	
	fmt.Println("\nComparison with paper results:")
	fmt.Printf("- Paper achieved ~98.9%% accuracy for spine datasets\n")
	fmt.Printf("- Paper achieved ~99.0%% accuracy for brain datasets\n")
	fmt.Printf("- Our implementation achieved %.2f%% accuracy\n", metrics.Accuracy)
	
	fmt.Println("\nParallel processing performance:")
	fmt.Printf("- Used %d cores for processing\n", *numCores)
	fmt.Printf("- Total processing time: %.2f seconds\n", processingTime.Seconds())
	fmt.Printf("- Paper reported ~70%% speedup with 8 cores vs single core\n")
	
	// Extract and save slices if requested (implements Algorithm 3 from the paper)
	if *extractSlices {
		fmt.Println("\nExtracting reconstructed slices along all axes...")
		
		// Get the reconstructed volume data
		volumeData, width, height, depth := reconstructor.GetVolumeData()
		
		// Create viewer
		viewer := visualization.NewViewer(volumeData, width, height, depth, *sliceGap)
		
		// Create output directory
		slicesPath := filepath.Join(outputDir, *slicesDir)
		
		// Extract and save slices along each axis
		for _, axis := range []string{"x", "y", "z"} {
			axisDir := filepath.Join(slicesPath, axis)
			fmt.Printf("Saving %s-axis slices to: %s\n", axis, axisDir)
			
			if err := viewer.SaveSliceSequence(axis, axisDir); err != nil {
				log.Printf("Warning: Failed to save %s-axis slices: %v", axis, err)
			}
		}
		
		fmt.Println("Slice extraction completed!")
	}
	
	// Print information about intermediary results if saved
	if *saveIntermediary {
		fmt.Println("\nIntermediary results saved to:")
		fmt.Printf("%s\n", intermediaryPath)
		fmt.Println("The following stages were saved:")
		fmt.Println("- 01_original_slices: Original input slices")
		fmt.Println("- 02_denoised_slices: Slices after shearlet denoising")
		fmt.Println("- 03_divided_dataset: Dataset divided into quadrants")
		fmt.Println("- 04_processed_subvolumes: Sub-volumes after kriging interpolation")
		fmt.Println("- 05_merged_volume: Final merged volume slices")
	}
} 