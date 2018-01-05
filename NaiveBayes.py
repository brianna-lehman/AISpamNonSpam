import sys
import xlwt
import xlrd
import os.path

all_data = [[], [], [], [], []]
means = [0.10455,0.21301,0.28066,0.065425,0.31222,0.095901,0.11421,0.10529,0.090067,0.23941,0.059824,0.5417,0.09393,0.058626,0.049205,0.24885,0.14259,0.18474,1.6621,0.085577,0.80976,0.1212,0.10165,0.094269,0.5495,0.26538,0.7673,0.12484,0.098915,0.10285,0.064753,0.047048,0.097229,0.047835,0.10541,0.097477,0.13695,0.013201,0.078629,0.064834,0.043667,0.13234,0.046099,0.079196,0.30122,0.17982,0.0054445,0.031869,0.038575,0.13903,0.016976,0.26907,0.075811,0.044238,5.1915,52.173,283.29,0.39404]
NUM_OF_FEATURES = 57
NUM_OF_TOTAL_EXAMPLES = 4601
ZERO = .0014

def main():
	filename = sys.argv[1]
	dataset = open(filename, "r")

	line_num = 0;
	for line in dataset:
		line = line.strip('\n')
		str_example = line.split(',')
		example = [float(x) for x in str_example]
		fold_index = line_num % 5
		fold = all_data[fold_index]
		fold.append(example)
		line_num += 1

	dataset.close()
	prob_spreadsheet = os.path.exists("feature_prob.xls")

	if not prob_spreadsheet:
		filename = "feature_prob.xls"
		book = xlwt.Workbook(encoding="utf-8")
		sh = book.add_sheet("Sheet 1")
		setup_spreadsheet(sh)

	final_stats = {}

	print("Pos = non-spam")
	print("Neg = spam")

	print("Iteration\t# Pos Samples (Train)\t# Neg Samples (Train)\t# Pos Samples (Test)\t# Neg Samples (Test)")

	for i in range(0, 5):
		test_fold = all_data.pop(i)

		train_spam_prob, train_not_spam_prob, feature_probabilities, train_spam_count, train_not_spam_count = train(all_data)
		error_stats, test_spam_count, test_not_spam_count = test(test_fold, train_spam_prob, train_not_spam_prob, feature_probabilities)
		# print # of positive samples/# of negative samples for training/dev
		print_data_split(i+1, train_spam_count, train_not_spam_count, test_spam_count, test_not_spam_count)
		write_data_to_file(i+1, all_data, test_fold)
		if not prob_spreadsheet:
			write_probability_to_spreadsheet(i+1, feature_probabilities, sh)
		final_stats[i+1] = error_stats

		all_data.insert(i,test_fold)

	if not prob_spreadsheet:
		book.save(filename)
	print()
	final_report(final_stats)

def train(training_folds_list):
	number_of_examples = 0
	spam = 0
	less_than_mean_spam_count = [0 for i in range(0, NUM_OF_FEATURES)]
	less_than_mean_not_spam_count = [0 for i in range(0, NUM_OF_FEATURES)]
	all_probabilities = []

	for fold in training_folds_list:
		number_of_examples += len(fold)
		for example in fold:
			isSpam = example[-1] == 1
			if isSpam:
				spam += 1
			for index,feature in enumerate(example[:-1]):
				# count the number of features in these training folds that is less than the mean
				if feature <= means[index] and isSpam:
					less_than_mean_spam_count[index] += 1
				if feature <= means[index] and not isSpam:
					less_than_mean_not_spam_count[index] += 1

	not_spam = number_of_examples - spam
	less_than_mean_spam = []
	less_than_mean_not_spam = []
	greater_than_mean_spam = []
	greater_than_mean_not_spam = []

	# calculating probabilities for all features and adding them to list
	for i in range(0, NUM_OF_FEATURES):
		less_than_mean_spam_prob = float(less_than_mean_spam_count[i] / spam)
		less_than_mean_not_spam_prob = float(less_than_mean_not_spam_count[i] / not_spam)
		greater_than_mean_spam_prob = float((spam - less_than_mean_spam_count[i]) / spam)
		greater_than_mean_not_spam_prob = float((not_spam - less_than_mean_not_spam_count[i]) / not_spam)

		# add P(fi <= mui | spam)
		if less_than_mean_spam_prob == 0:
			less_than_mean_spam.append(ZERO)
		else:
			less_than_mean_spam.append(less_than_mean_spam_prob)

		# add P(fi <= mui | not spam)
		if less_than_mean_not_spam_prob == 0:
			less_than_mean_not_spam.append(ZERO)
		else:
			less_than_mean_not_spam.append(less_than_mean_not_spam_prob)

		# add P(fi > mui | spam)
		if greater_than_mean_spam_prob == 0:
			greater_than_mean_spam.append(ZERO)
		else:
			greater_than_mean_spam.append(greater_than_mean_spam_prob)

		# add P(fi > mui | not spam)
		if greater_than_mean_not_spam_prob == 0:
			greater_than_mean_not_spam.append(ZERO)
		else:
			greater_than_mean_not_spam.append(greater_than_mean_not_spam_prob)

	# list of lists
	# external list represents every feature that an email can have
	# internal lists contain 4 probabilities for each feature
	all_probabilities = [ \
							[less_than_mean_spam[i], greater_than_mean_spam[i], \
							less_than_mean_not_spam[i], greater_than_mean_not_spam[i]] \
							for i in range(0, NUM_OF_FEATURES) \
						]

	# return probablity of spam, probability of not spam, and probablility of each feature
	return float(spam/number_of_examples), float(not_spam/number_of_examples), all_probabilities, spam, not_spam

def test(fold, spam, not_spam, feature_probabilities):
	s_ns_predictions = []
	number_of_examples = 0
	false_pos = 0
	false_neg = 0
	spam_count = 0

	for example in fold:
		number_of_examples += 1
		example_actual_spam = example[-1] == 1
		if example_actual_spam:
			spam_count += 1
		example_predict_spam = False
		spam_prob_product = spam
		not_spam_prob_product = not_spam

		for i, feature in enumerate(example[:-1]):
			if feature <= means[i]:
				spam_prob_product *= feature_probabilities[i][0]
				not_spam_prob_product *= feature_probabilities[i][2]
			else:
				spam_prob_product *= feature_probabilities[i][1]
				not_spam_prob_product *= feature_probabilities[i][3]

		if spam_prob_product > not_spam_prob_product:
			example_predict_spam = True

		s_ns_predictions.append(example_predict_spam)

		if example_predict_spam == True and example_actual_spam == False:
			false_pos += 1
		elif example_predict_spam == False and example_actual_spam == True:
			false_neg += 1

	return [float(false_pos/number_of_examples), float(false_neg/number_of_examples), \
			float((false_pos+false_neg)/number_of_examples), false_pos, false_neg], spam_count, number_of_examples-spam_count

# puts the headers in the feature probabilities spreadsheet
def setup_spreadsheet(sh):

	sh.write(0,0,"iter")

	col = 1
	f_num = 1

	while f_num <= NUM_OF_FEATURES:
		f = str(f_num)
		sh.write(0,col,"Pr(F"+f+"<=mui | spam)")
		col += 1
		sh.write(0,col,"Pr(F"+f+">mui | spam)")
		col += 1
		sh.write(0,col,"Pr(F"+f+"<=mui | non-spam)")
		col += 1
		sh.write(0,col,"Pr(F"+f+">mui | non-spam)")
		col += 1
		f_num += 1

# put the 4 probabilities for each feature into an excel spreadsheet
def write_probability_to_spreadsheet(iter, arr, sh):
	sh.write(iter,0,str(iter))

	col = 1

	for feature in arr:
		for prob in feature:
			sh.write(iter,col,prob)
			col += 1

# create two files, putting training data in one and testing data in another
def write_data_to_file(iter_int, train, test):
	def write(fold, file):
		for example in fold:
			for i,f in enumerate(example):
				feature = str(f)
				if i != len(example) - 1:
					file.write(feature+",")
				else:
					file.write(feature)
			file.write("\n")

	iter = str(iter_int)

	if not os.path.exists(iter+"_Train.txt"):
		# create file named iter+"_Train.txt"
		train_file = open(iter+"_Train.txt", 'a')

		for fold in train:
			write(fold, train_file)

		train_file.close()

	if not os.path.exists(iter+"_Dev.txt"):
		# create file named iter+"_Dev.txt"
		test_file = open(iter+"_Dev.txt", 'a')

		write(test, test_file)

		test_file.close()

# print spam and not spam counts to console
def print_data_split(iter, train_spam, train_not_spam, test_spam, test_not_spam):
	print("%d\t\t%d\t\t\t%d\t\t\t%d\t\t\t%d" %(iter, train_not_spam, train_spam, test_not_spam, test_spam))

# stats = i: probability of false positive, probability of false negative, overall probability of error, false positive count, false negative count
def final_report(stats):
	total_false_pos = 0
	total_false_neg = 0
	total_overall = 0

	for iteration,values in stats.items():
		print("Fold_%d, %f, %f, %f" %(iteration, values[0], values[1], values[2]))
		total_false_pos += values[3]
		total_false_neg += values[4]

	total_overall = total_false_pos + total_false_neg
	avg_false_pos = float(total_false_pos/NUM_OF_TOTAL_EXAMPLES)
	avg_false_neg = float(total_false_neg/NUM_OF_TOTAL_EXAMPLES)
	avg_overall = float(total_overall/NUM_OF_TOTAL_EXAMPLES)
	print("Avg, %f, %f, %f" %(avg_false_pos, avg_false_neg, avg_overall))

if __name__ == '__main__':
	main()