def inspect_review_file(filepath:str) -> None:
    try:
        with open (filepath, "r") as f:
            content = f.read()
            #print(content)
            lines = content.splitlines()
            print(lines[0])
            print(len(lines))
    except FileNotFoundError:
        print("file not found")
    except Exception as e:
        print(f"An error occured:{type(e).__name__}:{e}")

inspect_review_file("sample_reviews.csv")
inspect_review_file("nope.csv")

# lines = content.splitlines:
import pandas as pd
print(pd.__version__)