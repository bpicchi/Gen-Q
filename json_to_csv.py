import json
import csv


def json_to_csv_br():
	with open('wise_all.json') as json_file:
		data = json.load(json_file)

	data_file = open('data_file_br_50k.csv', 'w')
	csv_writer = csv.writer(data_file)
	categories = data.keys()

	category_map = {}
	index = 0
	for cat in categories:
		category_map.update({cat: index})
		index += 1

	headers = []
	headers.extend(('category', 'cite', 'quote'))
	headers.extend(categories)

	csv_writer.writerow(headers)

	num_rows = 0

	for category in categories:
		for quote_block in data[category]:
			cite = quote_block['cite']
			quote = quote_block['quote']

			if num_rows == 50000: return

			try:
				row = [category, cite, quote]
				category_index = category_map.get(category)
				for idx, cat in enumerate(categories):
					if idx == category_index:
						row.append(1)
					else:
						row.append(0)
				csv_writer.writerow(row)
				num_rows += 1

			except Exception as e:
				csv_writer.writerow([category, 'error', 'error'])

def json_to_csv(): 
	with open('wise_all.json') as json_file:
		data = json.load(json_file)

	data_file = open('data_file.csv', 'w')
	csv_writer = csv.writer(data_file)
	categories = data.keys()

	csv_writer.writerow(['category', 'cite', 'quote'])

	for cat in categories:
		for quote_block in data[cat]:
			cite = quote_block["cite"]
			quote = quote_block["quote"]
			try:
				csv_writer.writerow([cat, cite, quote])
			except Exception as e:
				csv_writer.writerow([cat, 'error', 'error'])

	data_file.close()



def main():
	json_to_csv()
	# json_to_csv_br()






if __name__ == "__main__":
	main()