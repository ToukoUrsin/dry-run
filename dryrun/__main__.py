"""Command line entry point for the real MuJoCo benchmark."""
import sys
from .benchmark import main
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
        sys.argv.pop(1)
    main()
