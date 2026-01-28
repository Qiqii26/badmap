//!     Bad Map - Bad apple video generator using real map data
//!     Copyright (C) 2026 The badmap authors
//!     This program is free software: you can redistribute it and/or modify
//!     it under the terms of the GNU General Public License as published by
//!     the Free Software Foundation, either version 3 of the License, or
//!     (at your option) any later version.
//!
//!     This program is distributed in the hope that it will be useful,
//!     but WITHOUT ANY WARRANTY; without even the implied warranty of
//!     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//!     GNU General Public License for more details.
//!
//!     You should have received a copy of the GNU General Public License
//!     along with this program.  If not, see <https://www.gnu.org/licenses/>.

use anyhow::{Context, Result};
use clap::Parser;
use image::{GrayImage, Luma};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use rand::Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use shapefile::{Reader, Shape};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use std::sync::Arc;

extern crate ffmpeg_next as ffmpeg;

// Configs, should be ajusted.
// for output width and height I would suggest to match the original
// video. The now configs uses 5gb ram when Pre-computing map views
const OUTPUT_WIDTH: u32 = 1280; 
const OUTPUT_HEIGHT: u32 = 720;
const RENDER_WIDTH: u32 = 640;
const RENDER_HEIGHT: u32 = 360;
const SIGNATURE_SIZE: usize = 64;
const BLUR_SIGMA: f32 = 2.0;
const THRESHOLD: f32 = 0.4;
const EDGE_WEIGHT: f32 = 0.6;

/// CLI Arguments
#[derive(Parser, Debug)]
#[command(name = "badmap")]
#[command(about = "Bad Apple Map Renderer")]
struct Args {
    /// Path to Bad Apple video
    #[arg(long, default_value = "badapple.mp4")]
    video: PathBuf,

    /// Path to Natural Earth vector data, best do git clone
    #[arg(long, default_value = "natural-earth-vector")]
    map: PathBuf,

    /// Output directory for frames
    #[arg(long, default_value = "frames_rust")]
    out: PathBuf,

    /// Start frame index
    #[arg(long, default_value_t = 0)]
    start: u32,

    /// End frame index
    #[arg(long, default_value_t = 0)]
    end: u32,

    /// Cache file for pre-computed views so you can resume
    #[arg(long, default_value = "cache_rust.bin")]
    cache: PathBuf,

    /// Number of view samples per zoom level per layer config
    #[arg(long, default_value_t = 1000)]
    samples: usize,

    /// Number of zoom levels
    #[arg(long, default_value_t = 20)]
    levels: usize,

    /// Number of thread
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Width for view precomputation
    #[arg(long, default_value_t = RENDER_WIDTH)]
    render_width: u32,

    /// Height for view precomputation
    #[arg(long, default_value_t = RENDER_HEIGHT)]
    render_height: u32,

    // --- Below Layer control ---
    /// Disable coastlines
    #[arg(long)]
    no_coastlines: bool,

    /// Disable rivers
    #[arg(long)]
    no_rivers: bool,

    /// Disable roads
    #[arg(long)]
    no_roads: bool,

    /// Disable country/state borders
    #[arg(long)]
    no_borders: bool,

    /// Disable lake outlines
    #[arg(long)]
    no_lakes: bool,

    /// Disable railroads
    #[arg(long)]
    no_railroads: bool,

    /// Invert colors (black/white swap)
    #[arg(long)]
    invert: bool,

    /// Only use coastlines (disable all other layers)
    #[arg(long)]
    coastlines_only: bool,
}

/// View parameters (geographic bounding box)
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ViewParams {
    min_lon: f64,
    max_lon: f64,
    min_lat: f64,
    max_lat: f64,
}

/// Pre-computed view with signature
#[derive(Clone, Serialize, Deserialize)]
struct PrecomputedView {
    params: ViewParams,
    signature: Vec<f32>,
    layer_config_id: u8, // Which layer configuration was used
}

/// Line geometry (simplified for performance)
#[derive(Clone)]
struct LineGeometry {
    points: Vec<(f64, f64)>, // (lon, lat) pairs
}

/// Polygon geometry
#[derive(Clone)]
struct PolygonGeometry {
    exterior: Vec<(f64, f64)>,
}

/// Layer configuration
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
struct LayerConfig {
    use_coastlines: bool,
    use_rivers: bool,
    use_roads: bool,
    use_borders: bool,
    use_lakes: bool,
    use_railroads: bool,
    invert_colors: bool,
}

impl LayerConfig {
    /// Get all preset layer configurations for variety. You can ajust them as need
    fn get_presets() -> Vec<LayerConfig> {
        vec![
            // Full layers - everything on
            LayerConfig { use_coastlines: true, use_rivers: true, use_roads: true, use_borders: true, use_lakes: true, use_railroads: true, invert_colors: false },
            // Coastlines only
            LayerConfig { use_coastlines: true, use_rivers: false, use_roads: false, use_borders: false, use_lakes: false, use_railroads: false, invert_colors: false },
            // Coastlines + rivers + lakes (natural features)
            LayerConfig { use_coastlines: true, use_rivers: true, use_roads: false, use_borders: false, use_lakes: true, use_railroads: false, invert_colors: false },
            // Roads and railroads only (infrastructure)
            LayerConfig { use_coastlines: false, use_rivers: false, use_roads: true, use_borders: false, use_lakes: false, use_railroads: true, invert_colors: false },
            // Borders only
            LayerConfig { use_coastlines: false, use_rivers: false, use_roads: false, use_borders: true, use_lakes: false, use_railroads: false, invert_colors: false },
            // Coastlines + borders (political + natural)
            LayerConfig { use_coastlines: true, use_rivers: false, use_roads: false, use_borders: true, use_lakes: false, use_railroads: false, invert_colors: false },
            // Inverted full layers
            LayerConfig { use_coastlines: true, use_rivers: true, use_roads: true, use_borders: true, use_lakes: true, use_railroads: true, invert_colors: true },
            // Inverted coastlines only
            LayerConfig { use_coastlines: true, use_rivers: false, use_roads: false, use_borders: false, use_lakes: false, use_railroads: false, invert_colors: true },
        ]
    }
}

/// Coastline/Map Database
struct CoastlineDatabase {
    coastlines: Vec<LineGeometry>,
    land_polygons: Vec<PolygonGeometry>,
    rivers: Vec<LineGeometry>,
    roads: Vec<LineGeometry>,
    borders: Vec<LineGeometry>,
    lakes: Vec<PolygonGeometry>,
    railroads: Vec<LineGeometry>,

    views: Vec<PrecomputedView>,
    layer_configs: Vec<LayerConfig>,

    render_width: u32,
    render_height: u32,
    output_width: u32,
    output_height: u32,

    layer_config: LayerConfig,
}

impl CoastlineDatabase {
    fn new(render_width: u32, render_height: u32) -> Self {
        let layer_configs = LayerConfig::get_presets();
        Self {
            coastlines: Vec::new(),
            land_polygons: Vec::new(),
            rivers: Vec::new(),
            roads: Vec::new(),
            borders: Vec::new(),
            lakes: Vec::new(),
            railroads: Vec::new(),
            views: Vec::new(),
            layer_configs,
            render_width,
            render_height,
            output_width: OUTPUT_WIDTH,
            output_height: OUTPUT_HEIGHT,
            layer_config: LayerConfig {
                use_coastlines: true,
                use_rivers: true,
                use_roads: true,
                use_borders: true,
                use_lakes: true,
                use_railroads: true,
                invert_colors: false,
            },
        }
    }

    fn load_geometry(&mut self, base_path: &Path) -> Result<()> {
        println!("Loading map geometrydash...");

        let physical_10m = base_path.join("10m_physical");
        let cultural_10m = base_path.join("10m_cultural");
        let physical_50m = base_path.join("50m_physical");
        let coastline_path = physical_10m.join("ne_10m_coastline.shp");
        if coastline_path.exists() {
            self.load_lines_into(&coastline_path, &mut self.coastlines.clone(), "coastlines")?;
        } else {
            let fallback = physical_50m.join("ne_50m_coastline.shp");
            if fallback.exists() {
                self.load_lines_into(&fallback, &mut self.coastlines.clone(), "coastlines (50m)")?;
            }
        }
        let land_path = physical_10m.join("ne_10m_land.shp");
        if land_path.exists() {
            self.load_polygons(&land_path, "land")?;
        }
        let rivers_path = physical_10m.join("ne_10m_rivers_lake_centerlines.shp");
        if rivers_path.exists() {
            let mut rivers = Vec::new();
            self.load_lines_into(&rivers_path, &mut rivers, "rivers")?;
            self.rivers = rivers;
        }
        let lakes_path = physical_10m.join("ne_10m_lakes.shp");
        if lakes_path.exists() {
            self.load_lake_polygons(&lakes_path, "lakes")?;
        }
        let roads_path = cultural_10m.join("ne_10m_roads.shp");
        if roads_path.exists() {
            let mut roads = Vec::new();
            self.load_lines_into(&roads_path, &mut roads, "roads")?;
            self.roads = roads;
        }
        let railroads_path = cultural_10m.join("ne_10m_railroads.shp");
        if railroads_path.exists() {
            let mut railroads = Vec::new();
            self.load_lines_into(&railroads_path, &mut railroads, "railroads")?;
            self.railroads = railroads;
        }
        let borders_path = cultural_10m.join("ne_10m_admin_0_boundary_lines_land.shp");
        if borders_path.exists() {
            let mut borders = Vec::new();
            self.load_lines_into(&borders_path, &mut borders, "country borders")?;
            self.borders = borders;
        }
        let state_borders_path = cultural_10m.join("ne_10m_admin_1_states_provinces_lines.shp");
        if state_borders_path.exists() {
            let mut state_borders = Vec::new();
            self.load_lines_into(&state_borders_path, &mut state_borders, "state borders")?;
            self.borders.extend(state_borders);
        }
        let coastline_path = physical_10m.join("ne_10m_coastline.shp");
        if coastline_path.exists() {
            let mut coastlines = Vec::new();
            self.load_lines_into(&coastline_path, &mut coastlines, "")?;
            self.coastlines = coastlines;
        }

        Ok(())
    }

    fn load_lines_into(
        &self,
        path: &Path,
        target: &mut Vec<LineGeometry>,
        name: &str,
    ) -> Result<()> {
        let mut reader = Reader::from_path(path).with_context(|| format!("Failed to open shapefile: {}", path.display()))?;
        let mut count = 0;
        for shape_record in reader.iter_shapes_and_records() {
            if let Ok((shape, _record)) = shape_record {
                match shape {
                    Shape::Polyline(polyline) => {
                        for part in polyline.parts() {
                            let points: Vec<(f64, f64)> =
                                part.iter().map(|p| (p.x, p.y)).collect();
                            if points.len() >= 2 {
                                target.push(LineGeometry { points });
                                count += 1;
                            }
                        }
                    }
                    Shape::PolylineZ(polyline) => {
                        for part in polyline.parts() {
                            let points: Vec<(f64, f64)> =
                                part.iter().map(|p| (p.x, p.y)).collect();
                            if points.len() >= 2 {
                                target.push(LineGeometry { points });
                                count += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        if !name.is_empty() {
            println!("Loaded {} {}", count, name);
        }
        Ok(())
    }

    fn load_polygons(&mut self, path: &Path, name: &str) -> Result<()> {
        let mut reader = Reader::from_path(path).with_context(|| format!("Failed to open shapefile: {}", path.display()))?;
        let mut count = 0;
        for shape_record in reader.iter_shapes_and_records() {
            if let Ok((shape, _record)) = shape_record {
                match shape {
                    Shape::Polygon(polygon) => {
                        for ring in polygon.rings() {
                            let exterior: Vec<(f64, f64)> =
                                ring.points().iter().map(|p| (p.x, p.y)).collect();
                            if exterior.len() >= 3 {
                                self.land_polygons.push(PolygonGeometry { exterior });
                                count += 1;
                            }
                        }
                    }
                    Shape::PolygonZ(polygon) => {
                        for ring in polygon.rings() {
                            let exterior: Vec<(f64, f64)> =
                                ring.points().iter().map(|p| (p.x, p.y)).collect();
                            if exterior.len() >= 3 {
                                self.land_polygons.push(PolygonGeometry { exterior });
                                count += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        println!("  Loaded {} {}", count, name);
        Ok(())
    }
    fn load_lake_polygons(&mut self, path: &Path, name: &str) -> Result<()> {
        let mut reader = Reader::from_path(path)
            .with_context(|| format!("Failed to open shapefile: {}", path.display()))?;

        let mut count = 0;
        for shape_record in reader.iter_shapes_and_records() {
            if let Ok((shape, _record)) = shape_record {
                match shape {
                    Shape::Polygon(polygon) => {
                        for ring in polygon.rings() {
                            let exterior: Vec<(f64, f64)> =
                                ring.points().iter().map(|p| (p.x, p.y)).collect();
                            if exterior.len() >= 3 {
                                self.lakes.push(PolygonGeometry { exterior });
                                count += 1;
                            }
                        }
                    }
                    Shape::PolygonZ(polygon) => {
                        for ring in polygon.rings() {
                            let exterior: Vec<(f64, f64)> =
                                ring.points().iter().map(|p| (p.x, p.y)).collect();
                            if exterior.len() >= 3 {
                                self.lakes.push(PolygonGeometry { exterior });
                                count += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        println!("Loaded {} {}", count, name);
        Ok(())
    }

    fn precompute_views(&mut self, num_levels: usize, samples_per_level: usize) {
        let layer_configs = LayerConfig::get_presets();
        let num_configs = layer_configs.len();
        let total_views = num_levels * samples_per_level * num_configs;
        
        println!(
            "Precomputing {} map views ({} levels x {} samples x {} layer configs at {}x{})...",
            total_views, num_levels, samples_per_level, num_configs, self.render_width, self.render_height
        );

        // Generate all view paras for all
        let mut rng = rand::thread_rng();
        let mut all_view_params: Vec<ViewParams> = Vec::with_capacity(num_levels * samples_per_level);

        for level in 0..num_levels {
            let zoom_factor = 1.5_f64.powi(level as i32); // Smoother zoom
            let lon_span = 360.0 / zoom_factor;
            let lat_span = 180.0 / zoom_factor;

            // Ensure valid range for sampling
            let lon_margin = (lon_span / 2.0).min(170.0);
            let lat_margin = (lat_span / 2.0).min(80.0);
            
            let lon_min = -180.0 + lon_margin;
            let lon_max = 180.0 - lon_margin;
            let lat_min = -90.0 + lat_margin;
            let lat_max = 90.0 - lat_margin;

            for _ in 0..samples_per_level {
                // Handle cases where span is too large for valid sampling
                let center_lon = if lon_min < lon_max {
                    rng.gen_range(lon_min..lon_max)
                } else {
                    rng.gen_range(-180.0..180.0)
                };
                let center_lat = if lat_min < lat_max {
                    rng.gen_range(lat_min..lat_max)
                } else {
                    rng.gen_range(-70.0..70.0) // Avoid extreme polar regions
                };

                all_view_params.push(ViewParams {
                    min_lon: (center_lon - lon_span / 2.0).max(-180.0),
                    max_lon: (center_lon + lon_span / 2.0).min(180.0),
                    min_lat: (center_lat - lat_span / 2.0).max(-90.0),
                    max_lat: (center_lat + lat_span / 2.0).min(90.0),
                });
            }
        }

        // Create all (view_params, layer_config_id) combinations
        let mut all_jobs: Vec<(ViewParams, u8)> = Vec::with_capacity(total_views);
        for params in &all_view_params {
            for (config_id, _) in layer_configs.iter().enumerate() {
                all_jobs.push((params.clone(), config_id as u8));
            }
        }

        // Pre-compute views in parallel
        let pb = ProgressBar::new(total_views as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        // Clone data needed for parallel rendering.
        let coastlines = self.coastlines.clone();
        let land_polygons = self.land_polygons.clone();
        let rivers = self.rivers.clone();
        let roads = self.roads.clone();
        let borders = self.borders.clone();
        let lakes = self.lakes.clone();
        let railroads = self.railroads.clone();
        let render_width = self.render_width;
        let render_height = self.render_height;

        let views: Vec<PrecomputedView> = all_jobs
            .par_iter()
            .progress_with(pb)
            .map(|(params, config_id)| {
                let config = &layer_configs[*config_id as usize];
                let rendered = render_view_internal(
                    params,
                    render_width,
                    render_height,
                    &coastlines,
                    &land_polygons,
                    &rivers,
                    &roads,
                    &borders,
                    &lakes,
                    &railroads,
                    config,
                );
                let signature = compute_signature(&rendered, render_width, render_height);
                PrecomputedView {
                    params: params.clone(),
                    signature,
                    layer_config_id: *config_id,
                }
            })
            .collect();

        self.views = views;
        self.layer_configs = layer_configs;
        println!("Total: {} views pre-computed", self.views.len());
    }

    fn find_best_match(&self, target_signature: &[f32]) -> Option<&PrecomputedView> {
        self.views.par_iter().min_by(|a, b| {
            let dist_a = hamming_distance(&a.signature, target_signature);
            let dist_b = hamming_distance(&b.signature, target_signature);
            dist_a.partial_cmp(&dist_b).unwrap()
        })
    }

    fn render_view_highres(&self, view: &PrecomputedView) -> GrayImage {
        let config = if (view.layer_config_id as usize) < self.layer_configs.len() {
            &self.layer_configs[view.layer_config_id as usize]
        } else {
            &self.layer_config
        };
        
        let rendered = render_view_internal(
            &view.params,
            self.output_width,
            self.output_height,
            &self.coastlines,
            &self.land_polygons,
            &self.rivers,
            &self.roads,
            &self.borders,
            &self.lakes,
            &self.railroads,
            config,
        );

        // Convert to GrayImage
        let mut img = GrayImage::new(self.output_width, self.output_height);
        for y in 0..self.output_height {
            for x in 0..self.output_width {
                let val = (rendered[(y * self.output_width + x) as usize] * 255.0) as u8;
                img.put_pixel(x, y, Luma([val]));
            }
        }
        img
    }

    fn save_cache(&self, path: &Path) -> Result<()> {
        println!("Saving cache to {}...", path.display());
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        // Save both views and layer configs
        bincode::serialize_into(writer, &(&self.views, &self.layer_configs))?;
        Ok(())
    }

    fn load_cache(&mut self, path: &Path) -> Result<()> {
        println!("Loading cached views from {}...", path.display());
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let (views, layer_configs): (Vec<PrecomputedView>, Vec<LayerConfig>) = bincode::deserialize_from(reader)?;
        self.views = views;
        self.layer_configs = layer_configs;
        println!("Loaded {} cached views with {} layer configs", self.views.len(), self.layer_configs.len());
        Ok(())
    }
}

/// Render a view of the map
fn render_view_internal(
    params: &ViewParams,
    width: u32,
    height: u32,
    coastlines: &[LineGeometry],
    land_polygons: &[PolygonGeometry],
    rivers: &[LineGeometry],
    roads: &[LineGeometry],
    borders: &[LineGeometry],
    lakes: &[PolygonGeometry],
    railroads: &[LineGeometry],
    config: &LayerConfig,
) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let mut land_mask = vec![0.0f32; w * h];
    let mut lines_img = vec![0.0f32; w * h];
    for poly in land_polygons {
        draw_filled_polygon(&mut land_mask, w, h, &poly.exterior, params);
    }
    if config.use_coastlines {
        for line in coastlines {
            draw_line_geometry(&mut lines_img, w, h, &line.points, params, 2);
        }
    }
    if config.use_rivers {
        for line in rivers {
            draw_line_geometry(&mut lines_img, w, h, &line.points, params, 1);
        }
    }
    if config.use_roads {
        for line in roads {
            draw_line_geometry(&mut lines_img, w, h, &line.points, params, 1);
        }
    }
    if config.use_railroads {
        for line in railroads {
            draw_line_geometry(&mut lines_img, w, h, &line.points, params, 1);
        }
    }
    if config.use_borders {
        for line in borders {
            draw_line_geometry(&mut lines_img, w, h, &line.points, params, 1);
        }
    }
    if config.use_lakes {
        for lake in lakes {
            draw_polygon_outline(&mut lines_img, w, h, &lake.exterior, params, 1);
        }
    }
    let mut result = vec![0.0f32; w * h];
    for i in 0..result.len() {
        result[i] = land_mask[i] * (1.0 - lines_img[i] * 0.8);
        if config.invert_colors {
            result[i] = 1.0 - result[i];
        }
    }

    result
}
fn draw_filled_polygon(
    buffer: &mut [f32],
    w: usize,
    h: usize,
    exterior: &[(f64, f64)],
    params: &ViewParams,
) {
    if exterior.len() < 3 {
        return;
    }

    // Convert to pixel coordinates
    let pixels: Vec<(i32, i32)> = exterior
        .iter()
        .map(|(lon, lat)| {
            let x = ((lon - params.min_lon) / (params.max_lon - params.min_lon) * w as f64) as i32;
            let y = ((params.max_lat - lat) / (params.max_lat - params.min_lat) * h as f64) as i32;
            (x, y)
        })
        .collect();

    // Find bounding box
    let min_y = pixels.iter().map(|(_, y)| *y).min().unwrap_or(0).max(0);
    let max_y = pixels
        .iter()
        .map(|(_, y)| *y)
        .max()
        .unwrap_or(0)
        .min(h as i32 - 1);

    // Scanline fill
    for y in min_y..=max_y {
        let mut intersections = Vec::new();

        for i in 0..pixels.len() {
            let (x1, y1) = pixels[i];
            let (x2, y2) = pixels[(i + 1) % pixels.len()];

            if (y1 <= y && y2 > y) || (y2 <= y && y1 > y) {
                let x = x1 + (y - y1) * (x2 - x1) / (y2 - y1).max(1);
                intersections.push(x);
            }
        }

        intersections.sort();

        for chunk in intersections.chunks(2) {
            if chunk.len() == 2 {
                let start_x = chunk[0].max(0) as usize;
                let end_x = (chunk[1].min(w as i32 - 1) as usize).min(w - 1);
                for x in start_x..=end_x {
                    buffer[y as usize * w + x] = 1.0;
                }
            }
        }
    }
}

/// Draw a line geometry using Bresenham's algorithm
fn draw_line_geometry(
    buffer: &mut [f32],
    w: usize,
    h: usize,
    points: &[(f64, f64)],
    params: &ViewParams,
    thickness: i32,
) {
    if points.len() < 2 {
        return;
    }

    for i in 0..points.len() - 1 {
        let x1 =
            ((points[i].0 - params.min_lon) / (params.max_lon - params.min_lon) * w as f64) as i32;
        let y1 =
            ((params.max_lat - points[i].1) / (params.max_lat - params.min_lat) * h as f64) as i32;
        let x2 = ((points[i + 1].0 - params.min_lon) / (params.max_lon - params.min_lon) * w as f64)
            as i32;
        let y2 = ((params.max_lat - points[i + 1].1) / (params.max_lat - params.min_lat) * h as f64)
            as i32;

        draw_line_bresenham(buffer, w, h, x1, y1, x2, y2, thickness);
    }
}

/// Draw polygon outline
fn draw_polygon_outline(
    buffer: &mut [f32],
    w: usize,
    h: usize,
    exterior: &[(f64, f64)],
    params: &ViewParams,
    thickness: i32,
) {
    if exterior.len() < 2 {
        return;
    }

    for i in 0..exterior.len() {
        let next = (i + 1) % exterior.len();
        let x1 = ((exterior[i].0 - params.min_lon) / (params.max_lon - params.min_lon) * w as f64)
            as i32;
        let y1 = ((params.max_lat - exterior[i].1) / (params.max_lat - params.min_lat) * h as f64)
            as i32;
        let x2 = ((exterior[next].0 - params.min_lon) / (params.max_lon - params.min_lon) * w as f64)
            as i32;
        let y2 = ((params.max_lat - exterior[next].1) / (params.max_lat - params.min_lat) * h as f64)
            as i32;

        draw_line_bresenham(buffer, w, h, x1, y1, x2, y2, thickness);
    }
}

/// Bresenham's line algorithm with thickness
fn draw_line_bresenham(
    buffer: &mut [f32],
    w: usize,
    h: usize,
    mut x1: i32,
    mut y1: i32,
    x2: i32,
    y2: i32,
    thickness: i32,
) {
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    let mut err = dx - dy;

    let half_t = thickness / 2;

    loop {
        // Draw pixel with thickness
        for dy_off in -half_t..=half_t {
            for dx_off in -half_t..=half_t {
                let px = x1 + dx_off;
                let py = y1 + dy_off;
                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    buffer[py as usize * w + px as usize] = 1.0;
                }
            }
        }

        if x1 == x2 && y1 == y2 {
            break;
        }

        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x1 += sx;
        }
        if e2 < dx {
            err += dx;
            y1 += sy;
        }
    }
}

/// Compute edge map using simple Sobel-like operator
fn compute_edges(image: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut edges = vec![0.0f32; w * h];
    
    for y in 1..h-1 {
        for x in 1..w-1 {
            // Sobel X
            let gx = image[(y-1) * w + (x+1)] - image[(y-1) * w + (x-1)]
                   + 2.0 * image[y * w + (x+1)] - 2.0 * image[y * w + (x-1)]
                   + image[(y+1) * w + (x+1)] - image[(y+1) * w + (x-1)];
            
            // Sobel Y
            let gy = image[(y+1) * w + (x-1)] - image[(y-1) * w + (x-1)]
                   + 2.0 * image[(y+1) * w + x] - 2.0 * image[(y-1) * w + x]
                   + image[(y+1) * w + (x+1)] - image[(y-1) * w + (x+1)];
            
            edges[y * w + x] = (gx * gx + gy * gy).sqrt().min(1.0);
        }
    }
    
    edges
}

/// Compute signature for shape matching (includes edge information)
fn compute_signature(image: &[f32], src_w: u32, src_h: u32) -> Vec<f32> {
    let sig_size = SIGNATURE_SIZE;
    let w = src_w as usize;
    let h = src_h as usize;
    
    // Compute edge map
    let edges = compute_edges(image, w, h);
    
    // Combined signature: intensity + edges
    let mut signature = vec![0.0f32; sig_size * sig_size * 2];

    let block_w = w / sig_size;
    let block_h = h / sig_size;

    for sy in 0..sig_size {
        for sx in 0..sig_size {
            let mut sum_intensity = 0.0;
            let mut sum_edge = 0.0;
            let mut count = 0;

            for by in 0..block_h {
                for bx in 0..block_w {
                    let src_x = sx * block_w + bx;
                    let src_y = sy * block_h + by;
                    if src_x < w && src_y < h {
                        sum_intensity += image[src_y * w + src_x];
                        sum_edge += edges[src_y * w + src_x];
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let avg_intensity = sum_intensity / count as f32;
                let avg_edge = sum_edge / count as f32;
                
                // First half: intensity signature (binary)
                signature[sy * sig_size + sx] = if avg_intensity > 0.4 { 1.0 } else { 0.0 };
                // Second half: edge signature (continuous for better matching)
                signature[sig_size * sig_size + sy * sig_size + sx] = avg_edge;
            }
        }
    }

    signature
}

/// Weighted distance between signatures (combines intensity and edge matching)
fn hamming_distance(a: &[f32], b: &[f32]) -> f32 {
    let half = a.len() / 2;
    let intensity_dist: f32 = a[..half].iter().zip(b[..half].iter())
        .map(|(x, y)| (x - y).abs())
        .sum();
    let edge_dist: f32 = a[half..].iter().zip(b[half..].iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt();
    
    // Weight edges more heavily for shape matching
    intensity_dist + EDGE_WEIGHT * edge_dist * (half as f32).sqrt()
}

/// Video frame extractor using ffmpeg
struct VideoExtractor {
    frames: Vec<GrayImage>,
    frame_count: u32,
    #[allow(dead_code)]
    fps: f64,
    #[allow(dead_code)]
    width: u32,
    #[allow(dead_code)]
    height: u32,
}

impl VideoExtractor {
    fn new(path: &Path) -> Result<Self> {
        ffmpeg::init()?;
        
        let mut ictx = ffmpeg::format::input(&path)
            .with_context(|| format!("Cannot open video: {}", path.display()))?;
        
        let input = ictx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;
        
        let video_stream_index = input.index();
        let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())?;
        let mut decoder = context_decoder.decoder().video()?;
        
        let width = decoder.width();
        let height = decoder.height();
        let fps = input.avg_frame_rate();
        let fps_f64 = fps.0 as f64 / fps.1 as f64;
        
        // Estimate frame count from duration
        let duration = ictx.duration() as f64 / ffmpeg::ffi::AV_TIME_BASE as f64;
        let frame_count = (duration * fps_f64) as u32;
        
        println!(
            "Video: ~{} frames, {:.2} FPS, {}x{}",
            frame_count, fps_f64, width, height
        );

        // Scaler to convert to grayscale
        let mut scaler = ffmpeg::software::scaling::context::Context::get(
            decoder.format(),
            width,
            height,
            ffmpeg::format::Pixel::GRAY8,
            width,
            height,
            ffmpeg::software::scaling::flag::Flags::BILINEAR,
        )?;

        let mut frames = Vec::new();
        let mut frame_count_actual = 0u64;

        println!("Decoding video frames...");
        let pb = ProgressBar::new(frame_count as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        for (stream, packet) in ictx.packets() {
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet)?;
                
                let mut decoded = ffmpeg::frame::Video::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    let mut gray_frame = ffmpeg::frame::Video::empty();
                    scaler.run(&decoded, &mut gray_frame)?;
                    
                    // Convert to GrayImage
                    let data = gray_frame.data(0);
                    let stride = gray_frame.stride(0);
                    let mut img = GrayImage::new(width, height);
                    
                    for y in 0..height {
                        for x in 0..width {
                            let pixel = data[(y as usize) * stride + (x as usize)];
                            img.put_pixel(x, y, Luma([pixel]));
                        }
                    }
                    frames.push(img);
                    frame_count_actual += 1;
                }
                pb.set_position(frame_count_actual);
            }
        }
        
        // Flush remaining frames
        decoder.send_eof()?;
        let mut decoded = ffmpeg::frame::Video::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut gray_frame = ffmpeg::frame::Video::empty();
            scaler.run(&decoded, &mut gray_frame)?;
            
            let data = gray_frame.data(0);
            let stride = gray_frame.stride(0);
            let mut img = GrayImage::new(width, height);
            
            for y in 0..height {
                for x in 0..width {
                    let pixel = data[(y as usize) * stride + (x as usize)];
                    img.put_pixel(x, y, Luma([pixel]));
                }
            }
            frames.push(img);
        }
        pb.finish_with_message("Decoding complete");

        let actual_frame_count = frames.len() as u32;
        println!("Decoded {} frames", actual_frame_count);

        Ok(Self {
            frames,
            frame_count: actual_frame_count,
            fps: fps_f64,
            width,
            height,
        })
    }

    fn get_frame_silhouette(&self, frame_idx: u32) -> Result<Vec<f32>> {
        if frame_idx as usize >= self.frames.len() {
            return Ok(vec![1.0; (OUTPUT_WIDTH * OUTPUT_HEIGHT) as usize]);
        }

        let frame = &self.frames[frame_idx as usize];
        
        // Resize to output dimensions
        let resized = image::imageops::resize(
            frame,
            OUTPUT_WIDTH,
            OUTPUT_HEIGHT,
            image::imageops::FilterType::Triangle,
        );

        // Apply simple box blur (approximation of Gaussian)
        let blurred = imageproc::filter::gaussian_blur_f32(&resized, BLUR_SIGMA);

        // Convert to float and threshold
        let mut result = Vec::with_capacity((OUTPUT_WIDTH * OUTPUT_HEIGHT) as usize);
        for pixel in blurred.pixels() {
            let val = pixel.0[0] as f32 / 255.0;
            result.push(if val >= THRESHOLD { 1.0 } else { 0.0 });
        }

        Ok(result)
    }

    fn get_frame_signature(&self, frame_idx: u32) -> Result<Vec<f32>> {
        if frame_idx as usize >= self.frames.len() {
            return Ok(vec![1.0; SIGNATURE_SIZE * SIGNATURE_SIZE]);
        }

        let frame = &self.frames[frame_idx as usize];
        
        // Resize to signature size
        let resized = image::imageops::resize(
            frame,
            SIGNATURE_SIZE as u32,
            SIGNATURE_SIZE as u32,
            image::imageops::FilterType::Triangle,
        );

        // Convert to binary signature
        let mut result = Vec::with_capacity(SIGNATURE_SIZE * SIGNATURE_SIZE);
        for pixel in resized.pixels() {
            let val = pixel.0[0] as f32 / 255.0;
            result.push(if val >= THRESHOLD { 1.0 } else { 0.0 });
        }

        Ok(result)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Setup thread pool
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()?;
    }

    // Create output directory
    fs::create_dir_all(&args.out)?;

    // Initialize video extractor
    let extractor = VideoExtractor::new(&args.video)?;
    let total_frames = extractor.frame_count;

    let start_frame = args.start;
    let end_frame = if args.end == 0 {
        total_frames
    } else {
        args.end
    };

    println!("Rendering frames {} to {}", start_frame, end_frame);

    // Initialize coastline database
    let mut coastline_db = CoastlineDatabase::new(args.render_width, args.render_height);
    coastline_db.output_width = OUTPUT_WIDTH;
    coastline_db.output_height = OUTPUT_HEIGHT;

    // Apply layer config
    if args.coastlines_only {
        coastline_db.layer_config.use_rivers = false;
        coastline_db.layer_config.use_roads = false;
        coastline_db.layer_config.use_borders = false;
        coastline_db.layer_config.use_lakes = false;
        coastline_db.layer_config.use_railroads = false;
    } else {
        coastline_db.layer_config.use_coastlines = !args.no_coastlines;
        coastline_db.layer_config.use_rivers = !args.no_rivers;
        coastline_db.layer_config.use_roads = !args.no_roads;
        coastline_db.layer_config.use_borders = !args.no_borders;
        coastline_db.layer_config.use_lakes = !args.no_lakes;
        coastline_db.layer_config.use_railroads = !args.no_railroads;
    }
    coastline_db.layer_config.invert_colors = args.invert;

    // Load or compute views
    if args.cache.exists() {
        coastline_db.load_cache(&args.cache)?;
        coastline_db.load_geometry(&args.map)?;
    } else {
        coastline_db.load_geometry(&args.map)?;
        coastline_db.precompute_views(args.levels, args.samples);
        coastline_db.save_cache(&args.cache)?;
    }

    println!("\nStarting render...");
    println!("{}", "=".repeat(60));
    println!("  - Pre-computed {} views of the world map", coastline_db.views.len());
    println!("  - Using {} different layer configurations", coastline_db.layer_configs.len());
    println!("{}", "-".repeat(60));
    println!("Layer configurations available:");
    for (i, config) in coastline_db.layer_configs.iter().enumerate() {
        println!("  Config #{}: coast={} rivers={} roads={} borders={} lakes={} rail={} inv={}",
            i,
            if config.use_coastlines { "Y" } else { "N" },
            if config.use_rivers { "Y" } else { "N" },
            if config.use_roads { "Y" } else { "N" },
            if config.use_borders { "Y" } else { "N" },
            if config.use_lakes { "Y" } else { "N" },
            if config.use_railroads { "Y" } else { "N" },
            if config.invert_colors { "Y" } else { "N" },
        );
    }
    println!("{}", "=".repeat(60));

    // Pre-extract all signatures for parallel frame rendering
    println!("Extracting video signatures...");
    let pb = ProgressBar::new((end_frame - start_frame) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    let mut signatures: Vec<(u32, Vec<f32>, Vec<f32>)> = Vec::new();
    for frame_idx in start_frame..end_frame {
        let signature = extractor.get_frame_signature(frame_idx)?;
        let silhouette = extractor.get_frame_silhouette(frame_idx)?;
        signatures.push((frame_idx, signature, silhouette));
        pb.inc(1);
    }
    pb.finish_with_message("Signatures extracted");

    // Render frames in parallel
    println!("\nRendering frames...");
    let pb = ProgressBar::new((end_frame - start_frame) as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    // Wrap database in Arc for parallel access
    let db = Arc::new(coastline_db);
    let out_path = args.out.clone();

    signatures
        .par_iter()
        .progress_with(pb.clone())
        .for_each(|(frame_idx, signature, silhouette)| {
            // Find best matching view
            if let Some(best_match) = db.find_best_match(signature) {
                let rendered = db.render_view_highres(best_match);
                let blend = 0.3f32;
                let mut final_img = GrayImage::new(OUTPUT_WIDTH, OUTPUT_HEIGHT);

                for y in 0..OUTPUT_HEIGHT {
                    for x in 0..OUTPUT_WIDTH {
                        let idx = (y * OUTPUT_WIDTH + x) as usize;
                        let map_val = rendered.get_pixel(x, y).0[0] as f32 / 255.0;
                        let sil_val = silhouette[idx];
                        let blended = (1.0 - blend) * map_val + blend * sil_val;
                        final_img.put_pixel(x, y, Luma([(blended * 255.0) as u8]));
                    }
                }

                // Save frame
                let output_path = out_path.join(format!("frame_{:06}.png", frame_idx));
                final_img.save(&output_path).ok();
            }
        });

    pb.finish_with_message("Rendering complete");

    println!("\nDone. Frames saved to {}/", args.out.display());
    println!("Layer configurations used: {} different presets", db.layer_configs.len());

    Ok(())
}
