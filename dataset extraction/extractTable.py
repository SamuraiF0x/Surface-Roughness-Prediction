import json
from docx import Document

# Open the Word document
document = Document('labele.docx')

# Get the first table in the document
table = document.tables[0]

# Get the column names from the first row of the table
keys = [cell.text.strip() for cell in table.rows[0].cells]

# Initialize an empty list to hold the table data
data = []

# Loop through the rows of the table
for row in table.rows[1:]:
    # Initialize an empty dictionary to hold the row data
    row_data = {}
    # Loop through the cells of the row and add them to the row data dictionary
    for idx, cell in enumerate(row.cells):
        row_data[keys[idx]] = cell.text.strip()
    # Add the row data to the list of table data
    data.append(row_data)

# Write the table data to a JSON file
with open('table_data.json', 'w') as outfile:
    json.dump(data, outfile)
