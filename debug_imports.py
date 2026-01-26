
import sys
import os

try:
    import blood
    print(f"blood imported from: {blood.__file__}")
    print(f"blood dir: {dir(blood)}")
    
    try:
        from blood import dataset
        print(f"from blood import dataset: SUCCESS. dir(dataset): {dir(dataset)}")
    except ImportError as e:
        print(f"from blood import dataset: FAILED. {e}")

    try:
        import blood.dataset
        print("import blood.dataset: SUCCESS")
    except ImportError as e:
        print(f"import blood.dataset: FAILED. {e}")

except ImportError as e:
    print(f"import blood: FAILED. {e}")
