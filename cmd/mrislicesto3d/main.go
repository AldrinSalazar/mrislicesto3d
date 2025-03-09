package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
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
	verbose := flag.Bool("verbose", false, "Enable verbose logging output")
	testEdgeThresholds := flag.Bool("test-edge-thresholds", false, "Test different edge detection thresholds on sample slices")
	thresholdValues := flag.String("thresholds", "0.1,0.2,0.3,0.4,0.5", "Comma-separated list of threshold values to test")
	edgeOutputDir := flag.String("edge-output-dir", "edge_threshold_test", "Directory to save edge detection test results")
	isoLevelPercent := flag.Float64("isolevel", 0.25, "IsoLevel percent for volume generation (0.0-1.0, default: 0.25)")
	edgeDetectionThreshold := flag.Float64("edge-threshold", 0.5, "Edge detection threshold (0.0-1.0, default: 0.5)")
	flag.Parse()

	// Validate inputs
	if *inputDir == "" {
		fmt.Println("Usage: mrislicesto3d -input <input_directory> [options]")
		fmt.Println("\nOptions:")
		fmt.Println("  -input <dir>             Directory containing 2D MRI slices (required)")
		fmt.Println("  -output <file>           Output STL filename (default: output.stl)")
		fmt.Println("  -cores <num>             Number of CPU cores to use (default: all available)")
		fmt.Println("  -gap <mm>                Inter-slice gap in mm (default: 1.5)")
		fmt.Println("  -extract-slices          Extract and save reconstructed slices along all axes")
		fmt.Println("  -slices-dir <dir>        Directory to save extracted slices (default: reconstructed_slices)")
		fmt.Println("  -save-intermediary       Save intermediary results during processing (default: true)")
		fmt.Println("  -intermediary-dir <dir>  Directory to save intermediary results (default: intermediary_results)")
		fmt.Println("  -verbose                 Enable verbose logging output")
		fmt.Println("  -test-edge-thresholds    Test different edge detection thresholds on sample slices")
		fmt.Println("  -thresholds <list>       Comma-separated list of threshold values to test (default: 0.1,0.2,0.3,0.4,0.5)")
		fmt.Println("  -edge-output-dir <dir>   Directory to save edge detection test results (default: edge_threshold_test)")
		fmt.Println("  -isolevel <value>        IsoLevel percent for volume generation (0.0-1.0, default: 0.25)")
		fmt.Println("  -edge-threshold <value>  Edge detection threshold (0.0-1.0, default: 0.5)")
		flag.Usage()
		os.Exit(1)
	}

	// Validate parameters
	if *isoLevelPercent < 0.0 || *isoLevelPercent > 1.0 {
		fmt.Printf("Error: isolevel must be between 0.0 and 1.0, got %.2f\n", *isoLevelPercent)
		os.Exit(1)
	}
	
	if *edgeDetectionThreshold < 0.0 || *edgeDetectionThreshold > 1.0 {
		fmt.Printf("Error: edge-threshold must be between 0.0 and 1.0, got %.2f\n", *edgeDetectionThreshold)
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
		Verbose:                *verbose,
		IsoLevelPercent:        *isoLevelPercent,
		EdgeDetectionThreshold: *edgeDetectionThreshold,
	}

	// Create reconstructor instance
	reconstructor := reconstruction.NewReconstructor(params)
	
	// Test edge thresholds if requested
	if *testEdgeThresholds {
		fmt.Println("Testing edge detection thresholds on sample slices...")
		
		// Parse threshold values
		thresholds := parseThresholds(*thresholdValues)
		if len(thresholds) == 0 {
			log.Fatalf("No valid threshold values provided")
		}
		
		fmt.Printf("Using threshold values: %v\n", thresholds)
		
		// Create edge output directory path
		edgeOutputPath := filepath.Join(outputDir, *edgeOutputDir)
		
		// Run edge threshold test
		outputPaths, err := reconstructor.TestEdgeThresholds(thresholds, edgeOutputPath)
		if err != nil {
			log.Fatalf("Edge threshold testing failed: %v", err)
		}
		
		fmt.Printf("Edge detection test completed successfully!\n")
		fmt.Printf("Results saved to: %s\n", edgeOutputPath)
		fmt.Printf("Generated %d images for comparison\n", len(outputPaths))
		
		// Exit after testing if no further processing is needed
		if !*extractSlices {
			os.Exit(0)
		}
	}

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
	
	fmt.Println("\nReconstruction parameters used:")
	fmt.Printf("- IsoLevel: %.2f\n", *isoLevelPercent)
	fmt.Printf("- Edge Detection Threshold: %.2f\n", *edgeDetectionThreshold)
}

// parseThresholds parses a comma-separated string of threshold values
func parseThresholds(thresholdsStr string) []float64 {
	parts := strings.Split(thresholdsStr, ",")
	thresholds := make([]float64, 0, len(parts))
	
	for _, part := range parts {
		value, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err == nil && value >= 0 && value <= 1 {
			thresholds = append(thresholds, value)
		}
	}
	
	return thresholds
}