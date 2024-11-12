import shutil, csv

arr = []
with open('Melanoma_detection\\Dataset\\assoc.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in spamreader:
        arr.append(', '.join(row).split(','))

for i in range(len(arr)):
    try:
        shutil.copyfile(f"Melanoma_detection\\Dataset\\Dataset\\Big_dir\\{arr[i][0]}.jpg", f"Melanoma_detection\\Dataset\\Dataset\\{arr[i][1][1:]}\\{arr[i][0]}.jpg")
    except FileNotFoundError:
        print(f"file {i} not found")
        