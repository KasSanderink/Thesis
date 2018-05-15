import os
import sys
import pandas as pd

def main():
    args = sys.argv

    main_path = os.getcwd() + '/' + args[2] + '.csv'
    new_path = os.getcwd() + '/' + args[1] + '.csv'

    main = pd.read_csv(main_path)
    new = pd.read_csv(new_path)

    try:
        new = new.drop(['target', 'year'], axis=1)
    except ValueError:
        pass

    if new.size > main.size:
        print("This is probably wrong.")
        return 1

    result = pd.concat([new, main], axis=1)
    # result.to_csv(args[1] + '.csv', index=False)
    result.to_csv('merged.csv', index=False)
    print("Succes!")
    return 0

main()