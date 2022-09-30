import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd

# Search for .csv and .tsv files
def browseFiles():
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select a File",
                                          filetypes=(("CSV files",
                                                      "*.csv*"), ("TSV files",
                                                                  "*.tsv*"),
                                                     ("all files",
                                                      "*.*")))
    if filename:
        data = pd.read_csv(filename)
        print(data)

    header.configure(text="File Opened: " + filename)


# Initialize GUI
root = tk.Tk()

# Window title
root.title("Gaussian Mixture Model")

# Background color
root.config(bg='#3f3f3f')

# Window dimensions
width = int(root.winfo_screenwidth() / 2 - 300)
height = int(root.winfo_screenheight() / 2 - 200)
root.geometry(f'{600}x{400}+{width}+{height}')

# Info text for initial window
header = tk.Label(root, text='Generate a Gaussian Mixture Model for given data, accepts .csv or .tsv files',
                  bg='#3f3f3f', fg='#f0dfaf', padx=10, pady=10, font=15)
header.place(relx=0.5, rely=0.1, anchor='center')

# Browse file button
browse = tk.Button(root, text='Browse Files', command=browseFiles,
                   bg='#323232', fg='#f0dfaf', padx=10, pady=10, font=15)
browse.place(relx=0.5, rely=0.5, anchor='center')

# Call GMM using input file


root.mainloop()
