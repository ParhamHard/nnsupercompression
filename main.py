#!/usr/bin/env python3
"""
Main entry point for Neural Network Supercompression Library.

This script acts as the interface to the project and can be run from the command line.
"""

import sys
import argparse
from src.cli import compress_text, display_results
from examples.image_compression import compress_image_to_1kb


def main():
    """Main entry point for the project."""
    parser = argparse.ArgumentParser(
        description="Neural Network Supercompression Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress text (basic method)
  python main.py text "Hello, World!"
  
  # Compress text (perfect method)
  python main.py text "Hello, World!" --method perfect
  
  # Compress image
  python main.py image assets/Morden_army_ensign.png
  
  # Compress image with custom output
  python main.py image assets/Morden_army_ensign.png --output output.png
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Text compression subcommand
    text_parser = subparsers.add_parser('text', help='Compress text')
    text_parser.add_argument('text', help='Text to compress')
    text_parser.add_argument('--method', choices=['basic', 'perfect'], default='basic',
                            help='Compression method (default: basic)')
    text_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    
    # Image compression subcommand
    image_parser = subparsers.add_parser('image', help='Compress image')
    image_parser.add_argument('image_path', help='Path to image file')
    image_parser.add_argument('--output', help='Output path for reconstructed image')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'text':
            results = compress_text(args.text, args.method, args.epochs)
            display_results(results)
            return 0
        elif args.command == 'image':
            compress_image_to_1kb(args.image_path, args.output)
            return 0
        else:
            parser.print_help()
            return 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

