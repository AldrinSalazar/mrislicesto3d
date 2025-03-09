package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"mrislicesto3d/pkg/config"
	"mrislicesto3d/pkg/reconstruction"
	"mrislicesto3d/pkg/visualization"
)

func main() {
	// Parse command line arguments
	inputDir := flag.String("input", "", "Directory containing 2D MRI slices")
	outputName := flag.String("output", "output.stl", "Output STL filename")
	configFile := flag.String("config", "config.yaml", "Path to configuration file")
	verbose := flag.Bool("verbose", false, "Enable verbose logging output")
	createDefaultConfig := flag.Bool("create-config", false, "Create a default configuration file if it doesn't exist")
	flag.Parse()

	// Create default config file if requested
	if *createDefaultConfig {
		defaultConfigPath := *configFile
		if err := config.CreateDefaultConfigFile(defaultConfigPath); err != nil {
			log.Fatalf("Failed to create default configuration: %v", err)
		}
		fmt.Printf("Default configuration file created at: %s\n", defaultConfigPath)
		if *inputDir == "" {
			// Exit if only creating config file
			os.Exit(0)
		}
	}

	// Load configuration
	cfg, err := config.LoadConfig(*configFile)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	// Validate inputs
	if *inputDir == "" {
		fmt.Println("Usage: mrislicesto3d -input <input_directory> [options]")
		fmt.Println("\nOptions:")
		fmt.Println("  -input <dir>             Directory containing 2D MRI slices (required)")
		fmt.Println("  -output <file>           Output STL filename (default: output.stl)")
		fmt.Println("  -config <file>           Path to configuration file (default: config.yaml)")
		fmt.Println("  -verbose                 Enable verbose logging output")
		fmt.Println("  -create-config           Create a default configuration file if it doesn't exist")
		fmt.Println("\nConfiguration parameters are stored in the YAML config file.")
		fmt.Println("Run with -create-config to generate a default configuration file.")
		flag.Usage()
		os.Exit(1)
	}

	// Validate parameters
	if cfg.Processing.IsoLevelPercent < 0.0 || cfg.Processing.IsoLevelPercent > 1.0 {
		fmt.Printf("Error: isolevel must be between 0.0 and 1.0, got %.2f\n", cfg.Processing.IsoLevelPercent)
		os.Exit(1)
	}

	if cfg.Processing.EdgeDetectionThreshold < 0.0 || cfg.Processing.EdgeDetectionThreshold > 1.0 {
		fmt.Printf("Error: edge-threshold must be between 0.0 and 1.0, got %.2f\n", cfg.Processing.EdgeDetectionThreshold)
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
	intermediaryPath := filepath.Join(outputDir, "intermediary_results")

	fmt.Println("================================")
	fmt.Println("FAST 3D VOLUMETRIC IMAGE RECONSTRUCTION FROM 2D MRI SLICES BY PARALLEL PROCESSING")
	fmt.Println("Based on the paper by Somoballi Ghoshal et al.")
	fmt.Println("================================")

	// Initialize reconstruction parameters
	params := &reconstruction.Params{
		InputDir:                *inputDir,
		OutputFile:              outputPath,
		NumCores:                cfg.Processing.NumCores,
		SliceGap:                cfg.Processing.SliceGap,
		SaveIntermediaryResults: cfg.Output.SaveIntermediaryResults,
		IntermediaryDir:         intermediaryPath,
		Verbose:                 *verbose || cfg.Output.Verbose,
		IsoLevelPercent:         cfg.Processing.IsoLevelPercent,
		EdgeDetectionThreshold:  cfg.Processing.EdgeDetectionThreshold,
		ShearletScales:          cfg.Shearlet.Scales,
		ShearletShears:          cfg.Shearlet.Shears,
		ShearletConeParam:       cfg.Shearlet.ConeParam,
	}

	// If NumCores is 0, use all available cores
	if params.NumCores == 0 {
		params.NumCores = runtime.NumCPU()
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
	fmt.Printf("\nReconstruction completed successfully in %.2f seconds!\n", processingTime.Seconds())
	fmt.Printf("Output 3D model saved to: %s\n\n", outputPath)

	fmt.Println("\nParallel processing performance:")
	fmt.Printf("- Used %d cores for processing\n", params.NumCores)
	fmt.Printf("- Total processing time: %.2f seconds\n", processingTime.Seconds())
	fmt.Printf("- Paper reported ~70%% speedup with 8 cores vs single core\n")

	// Extract and save slices if requested (implements Algorithm 3 from the paper)
	extractSlices := false // Default to false as we removed this from config
	if extractSlices {
		fmt.Println("\nExtracting reconstructed slices along all axes...")

		// Get the reconstructed volume data
		volumeData, width, height, depth := reconstructor.GetVolumeData()

		// Create viewer
		viewer := visualization.NewViewer(volumeData, width, height, depth, cfg.Processing.SliceGap)

		// Create output directory
		slicesPath := filepath.Join(outputDir, "reconstructed_slices")

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
	if cfg.Output.SaveIntermediaryResults {
		fmt.Println("\nIntermediary results saved to:")
		fmt.Printf("- %s\n", intermediaryPath)
	}

	fmt.Println("\nThank you for using the MRI Slices to 3D reconstruction tool!")
}
