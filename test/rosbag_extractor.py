#!/usr/bin/env python3
"""
Extract RGB images from RealSense bag file to PNG files
Uses pyrealsense2 library (no ROS required)
"""
import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
import argparse

def extract_rgb_images(bag_path, output_dir, skip_frames=0):
    """
    Extract RGB images from RealSense bag file
    
    Args:
        bag_path: Path to the .bag file created by RealSense Viewer
        output_dir: Directory to save extracted PNG images
        skip_frames: Extract every Nth frame (0 = extract all frames)
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Opening RealSense bag: {bag_path}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Configure pipeline to read from bag file
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Tell config to read from the bag file
        rs.config.enable_device_from_file(config, bag_path, repeat_playback=False)
        
        # Enable color stream
        config.enable_stream(rs.stream.color)
        
        # Start pipeline
        profile = pipeline.start(config)
        
        # Get device from pipeline (for playback control)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)  # Process as fast as possible
        
        # Get stream info
        color_profile = profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        print(f"\nColor stream info:")
        print(f"  Resolution: {color_intrinsics.width}x{color_intrinsics.height}")
        print(f"  Format: {color_profile.format()}")
        print(f"  FPS: {color_profile.fps()}")
        
        print(f"\nExtracting RGB frames...")
        
        frame_count = 0
        saved_count = 0
        
        while True:
            try:
                # Wait for frames
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                
                # Get color frame
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                
                frame_count += 1
                
                # Skip frames if requested
                if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                    continue
                
                # Convert to numpy array (RealSense returns RGB)
                color_image = np.asanyarray(color_frame.get_data())
                
                # Convert RGB to BGR for OpenCV
                color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                # Get timestamp
                timestamp = color_frame.get_timestamp()
                frame_number = color_frame.get_frame_number()
                
                # Save as PNG
                filename = f"frame_{saved_count:06d}_ts{timestamp:.0f}_fn{frame_number}.png"
                filepath = output_path / filename
                
                cv2.imwrite(str(filepath), color_image_bgr)
                saved_count += 1
                
                if saved_count % 10 == 0:
                    print(f"  Saved {saved_count} images (processed {frame_count} frames)...", end='\r')
                
            except RuntimeError as e:
                # End of file or timeout
                if "Frame didn't arrive" in str(e):
                    print("\n  Reached end of bag file")
                    break
                else:
                    print(f"\n⚠ Error: {e}")
                    break
        
        print(f"\n✓ Successfully extracted {saved_count} RGB images")
        print(f"  Total frames in bag: {frame_count}")
        print(f"  Output directory: {output_dir}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.stop()

def get_bag_info(bag_path):
    """Print information about streams in the bag file"""
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        rs.config.enable_device_from_file(config, bag_path)
        
        profile = pipeline.start(config)
        device = profile.get_device()
        
        print(f"\nBag file information: {bag_path}")
        print("=" * 80)
        
        # Get playback device
        playback = device.as_playback()
        duration = playback.get_duration().total_seconds()
        print(f"Duration: {duration:.2f} seconds")
        
        # List all streams
        print("\nAvailable streams:")
        for i, stream_profile in enumerate(profile.get_streams()):
            stream = stream_profile.stream_type()
            fmt = stream_profile.format()
            
            if stream_profile.is_video_stream_profile():
                video_profile = stream_profile.as_video_stream_profile()
                intrinsics = video_profile.get_intrinsics()
                print(f"  [{i}] {stream} stream:")
                print(f"      Resolution: {intrinsics.width}x{intrinsics.height}")
                print(f"      Format: {fmt}")
                print(f"      FPS: {video_profile.fps()}")
            else:
                print(f"  [{i}] {stream} stream (Format: {fmt})")
        
        pipeline.stop()
        
    except Exception as e:
        print(f"Error reading bag info: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract RGB images from RealSense bag file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show bag file information
  python3 extract_rgb_from_realsense_bag.py recording.bag --info
  
  # Extract all RGB frames
  python3 extract_rgb_from_realsense_bag.py recording.bag -o ./rgb_images
  
  # Extract every 5th frame
  python3 extract_rgb_from_realsense_bag.py recording.bag -o ./rgb_images --skip 4
        """
    )
    
    parser.add_argument('bag_file', help='Path to RealSense .bag file')
    parser.add_argument('-o', '--output', default='./extracted_rgb',
                        help='Output directory for PNG files (default: ./extracted_rgb)')
    parser.add_argument('--skip', type=int, default=0,
                        help='Extract every Nth frame (e.g., --skip 4 saves every 5th frame)')
    parser.add_argument('--info', action='store_true',
                        help='Show bag file information and exit')
    
    args = parser.parse_args()
    
    if args.info:
        get_bag_info(args.bag_file)
    else:
        extract_rgb_images(args.bag_file, args.output, args.skip)