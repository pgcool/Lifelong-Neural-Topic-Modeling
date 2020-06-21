import sys, os
import numpy as np
import csv
import shutil


# Configure these parameters
target_data_dir = "TMNtitle" # ["20NSshort", "TMNtitle", "R21578title"]

source_data_dirs = ["20NS", "R21578", "TMN", "AGnews"]

new_data_dir = target_data_dir + "_" + "ALL" + "_target_vocab"



for source_data_dir in source_data_dirs:
	# Getting source data vocab
	source_vocab = []
	with open("./datasets/" + source_data_dir + "/vocab_docnade.vocab", "r") as f:
		for line in f.readlines():
			source_vocab.append(line.strip())

	# Getting target data vocab
	target_vocab = []
	with open("./datasets/" + target_data_dir + "/vocab_docnade.vocab", "r") as f:
		for line in f.readlines():
			target_vocab.append(line.strip())


	def transform_doc(doc, source_vocab, target_vocab):
		indices = doc.strip().split()
		not_found = True
		new_indices = []
		for index in indices:
			try:
				new_index = str(target_vocab.index(source_vocab[int(index.strip())]))
				not_found = False
			except ValueError:
				continue
			new_indices.append(new_index)

		if not_found:
			return None
		else:
			return " ".join(new_indices)


	# Converting Training, Validation and Test data indices according to common vocab

	if not os.path.exists("./new_datasets/" + new_data_dir + "/" + source_data_dir):
		os.makedirs("./new_datasets/" + new_data_dir + "/" + source_data_dir)


	# Converting Training dataset
	new_training_docs = []
	new_training_labels = []
	with open("./datasets/" + source_data_dir + "/training_docnade.csv", "r") as f_tr_in:
		csv_reader = csv.reader(f_tr_in, delimiter=',')
		for row in csv_reader:
			new_doc = transform_doc(row[1], source_vocab, target_vocab)
			if new_doc is None:
				continue
			else:
				new_training_docs.append(new_doc)
				new_training_labels.append(row[0])

	with open("./new_datasets/" + new_data_dir + "/" + source_data_dir + "/training_docnade.csv", "w") as f_tr_out:
		csv_writer = csv.writer(f_tr_out, delimiter=',')
		for new_label, new_doc in zip(new_training_labels, new_training_docs):
			csv_writer.writerow([new_label, new_doc])

	# Writing document indices
	with open("./new_datasets/" + new_data_dir + "/" + source_data_dir + "/training_document_indices.csv", "w") as f_tr_doc_ind_out:
		csv_writer = csv.writer(f_tr_doc_ind_out, delimiter=',')
		for index in range(len(new_training_docs)):
			csv_writer.writerow([str(0), str(index)])



	# Converting Validation dataset
	new_validation_docs = []
	new_validation_labels = []
	with open("./datasets/" + source_data_dir + "/validation_docnade.csv", "r") as f_va_in:
		csv_reader = csv.reader(f_va_in, delimiter=',')
		for row in csv_reader:
			new_doc = transform_doc(row[1], source_vocab, target_vocab)
			if new_doc is None:
				continue
			else:
				new_validation_docs.append(new_doc)
				new_validation_labels.append(row[0])

	with open("./new_datasets/" + new_data_dir + "/" + source_data_dir + "/validation_docnade.csv", "w") as f_va_out:
		csv_writer = csv.writer(f_va_out, delimiter=',')
		for new_label, new_doc in zip(new_validation_labels, new_validation_docs):
			csv_writer.writerow([new_label, new_doc])

	# Writing document indices
	with open("./new_datasets/" + new_data_dir + "/" + source_data_dir + "/validation_document_indices.csv", "w") as f_va_doc_ind_out:
		csv_writer = csv.writer(f_va_doc_ind_out, delimiter=',')
		for index in range(len(new_validation_docs)):
			csv_writer.writerow([str(0), str(index)])



	# Converting Test dataset
	new_test_docs = []
	new_test_labels = []
	with open("./datasets/" + source_data_dir + "/test_docnade.csv", "r") as f_te_in:
		csv_reader = csv.reader(f_te_in, delimiter=',')
		for row in csv_reader:
			new_doc = transform_doc(row[1], source_vocab, target_vocab)
			if new_doc is None:
				continue
			else:
				new_test_docs.append(new_doc)
				new_test_labels.append(row[0])

	with open("./new_datasets/" + new_data_dir + "/" + source_data_dir + "/test_docnade.csv", "w") as f_te_out:
		csv_writer = csv.writer(f_te_out, delimiter=',')
		for new_label, new_doc in zip(new_test_labels, new_test_docs):
			csv_writer.writerow([new_label, new_doc])

	# Writing document indices
	with open("./new_datasets/" + new_data_dir + "/" + source_data_dir + "/test_document_indices.csv", "w") as f_te_doc_ind_out:
		csv_writer = csv.writer(f_te_doc_ind_out, delimiter=',')
		for index in range(len(new_test_docs)):
			csv_writer.writerow([str(0), str(index)])


	# Writing common vocab
	with open("./new_datasets/" + new_data_dir + "/" + source_data_dir + "/vocab_docnade.vocab", "w") as f_vocab:
		f_vocab.write("\n".join(target_vocab))


# Copying target directory
shutil.copytree("./datasets/" + target_data_dir, "./new_datasets/" + new_data_dir + "/" + target_data_dir)