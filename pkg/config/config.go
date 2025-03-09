// Package config provides configuration loading and management for mrislicesto3d.
// It handles loading configuration from YAML files and provides default values.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"gopkg.in/yaml.v3"
)

// Config represents the application configuration loaded from YAML
type Config struct {
	// Processing parameters
	Processing struct {
		// NumCores specifies how many CPU cores to use for parallel processing
		NumCores int `yaml:"numCores"`
		
		// SliceGap represents the physical distance between consecutive MRI slices in mm
		SliceGap float64 `yaml:"sliceGap"`
		
		// IsoLevelPercent controls the threshold for final volume generation
		IsoLevelPercent float64 `yaml:"isoLevelPercent"`
		
		// EdgeDetectionThreshold controls the sensitivity of edge detection
		EdgeDetectionThreshold float64 `yaml:"edgeDetectionThreshold"`
	} `yaml:"processing"`
	
	// Shearlet transform parameters
	Shearlet struct {
		// Scales is the number of scales for the shearlet transform
		Scales int `yaml:"scales"`
		
		// Shears is the number of shears for the shearlet transform
		Shears int `yaml:"shears"`
		
		// ConeParam is the cone parameter for the shearlet transform
		ConeParam float64 `yaml:"coneParam"`
	} `yaml:"shearlet"`
	
	// Output parameters
	Output struct {
		// SaveIntermediaryResults determines whether to save intermediary processing results
		SaveIntermediaryResults bool `yaml:"saveIntermediaryResults"`
		
		// Verbose controls the level of logging output
		Verbose bool `yaml:"verbose"`
	} `yaml:"output"`
	
	// Test parameters
	Test struct {
		// EdgeThresholds is a list of thresholds to test for edge detection
		EdgeThresholds []float64 `yaml:"edgeThresholds"`
		
		// EdgeOutputDir is the directory to save edge detection test results
		EdgeOutputDir string `yaml:"edgeOutputDir"`
	} `yaml:"test"`
}

// DefaultConfig returns a configuration with default values
func DefaultConfig() *Config {
	cfg := &Config{}
	
	// Set default processing parameters
	cfg.Processing.NumCores = runtime.NumCPU() // Use all available cores by default
	cfg.Processing.SliceGap = 1.0
	cfg.Processing.IsoLevelPercent = 0.25
	cfg.Processing.EdgeDetectionThreshold = 0.5
	
	// Set default shearlet parameters
	cfg.Shearlet.Scales = 3
	cfg.Shearlet.Shears = 8
	cfg.Shearlet.ConeParam = 1.0
	
	// Set default output parameters
	cfg.Output.SaveIntermediaryResults = false
	cfg.Output.Verbose = true
	
	// Set default test parameters
	cfg.Test.EdgeThresholds = []float64{0.1, 0.3, 0.5, 0.7, 0.9}
	cfg.Test.EdgeOutputDir = "edge_threshold_test"
	
	return cfg
}

// LoadConfig loads configuration from a YAML file
// If the file doesn't exist, it returns the default configuration
func LoadConfig(configPath string) (*Config, error) {
	cfg := DefaultConfig()
	
	// Check if config file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		return cfg, nil
	}
	
	// Read config file
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("error reading config file: %w", err)
	}
	
	// Parse YAML
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("error parsing config file: %w", err)
	}
	
	return cfg, nil
}

// SaveConfig saves the configuration to a YAML file
func SaveConfig(cfg *Config, configPath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(configPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("error creating config directory: %w", err)
	}
	
	// Marshal config to YAML
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("error marshaling config: %w", err)
	}
	
	// Write to file
	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("error writing config file: %w", err)
	}
	
	return nil
}

// CreateDefaultConfigFile creates a default configuration file at the specified path
func CreateDefaultConfigFile(configPath string) error {
	cfg := DefaultConfig()
	return SaveConfig(cfg, configPath)
}
