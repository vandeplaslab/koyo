"""System utilities."""
import sys
import platform

IS_WIN = sys.platform == "win32"
IS_LINUX = sys.platform == "linux"
IS_MAC = sys.platform == "darwin"
IS_MAC_ARM = IS_MAC and platform.processor() == "arm"
