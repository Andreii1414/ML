import os
import math
from collections import defaultdict
import random
import matplotlib.pyplot as plt

def load_emails(directory):
    emails = []
    for filename in os.listdir(directory):
        if filename.startswith("spm"):
            with open(os.path.join(directory, filename), 'r', encoding='latin-1') as file:
                content = file.read()
                emails.append((content, 'spam'))
        else:
            with open(os.path.join(directory, filename), 'r', encoding='latin-1') as file:
                content = file.read()
                emails.append((content, 'regular'))
    return emails

def tokenize(email):
    return [word.strip(".,") for word in email.split()]

def train_naive_bayes(train_set):
    #la antrenare, calculam frecventa cuvintelor din fiecare clasa
    spam_word_counts = defaultdict(int)
    regular_word_counts = defaultdict(int)
    spam_total_words = 0
    regular_total_words = 0
    spam_email_count = 0
    regular_email_count = 0

    for email, label in train_set:
        words = tokenize(email)
        if label == 'spam':
            spam_email_count += 1
            for word in words:
                spam_word_counts[word] += 1
                spam_total_words += 1
        else:
            regular_email_count += 1
            for word in words:
                regular_word_counts[word] += 1
                regular_total_words += 1

    return spam_word_counts, regular_word_counts, spam_total_words, regular_total_words, spam_email_count, regular_email_count

def calculate_class_probability(word_counts, total_words, word):
    #calculam probabilitatea conditionata pentru un cuvant
    return (word_counts[word] + 1) / (total_words + 2)

def calculate_prior_probability(email_count, total_email_count):
    return email_count / total_email_count

def classify_naive_bayes(email, spam_word_counts, regular_word_counts, spam_total_words, regular_total_words, spam_email_count, regular_email_count):
    words = tokenize(email)
    spam_probability = math.log(calculate_prior_probability(spam_email_count, spam_email_count + regular_email_count))
    regular_probability = math.log(calculate_prior_probability(regular_email_count, spam_email_count + regular_email_count))

    for word in words:
        spam_probability += math.log(calculate_class_probability(spam_word_counts, spam_total_words, word))
        regular_probability += math.log(calculate_class_probability(regular_word_counts, regular_total_words, word))

    #Returnam clasa cu probabilitatea mai mare
    if spam_probability > regular_probability:
        return 'spam'
    else:
        return 'regular'

data_directory = "C:\\Users\\andrei14\\Downloads\\lingspam_public\\bare"

all_emails = []

#Selectam email-urile din primele 9 foldere pentru antrenare
for i in range(1, 10):
    folder_path = os.path.join(data_directory, f"part{i}")
    all_emails += load_emails(folder_path)

random.shuffle(all_emails)

#Antrenarea clasificatorului Naive Bayes
spam_word_counts, regular_word_counts, spam_total_words, regular_total_words, spam_email_count, regular_email_count = train_naive_bayes(all_emails)

#Selectam email-urile din folderul 10 pentru testare
test_folder_path = os.path.join(data_directory, "part10")
test_set = load_emails(test_folder_path)

#Testarea clasificatorului Naive Bayes
correct_predictions = 0
for email, label in test_set:
    predicted_label = classify_naive_bayes(email, spam_word_counts, regular_word_counts, spam_total_words, regular_total_words, spam_email_count, regular_email_count)
    if predicted_label == label:
        correct_predictions += 1

accuracy = correct_predictions / len(test_set)
print("Acuratete: ", accuracy)

def loocv(data_directory):
    accuracies = []
    all_emails = []

    for i in range(1, 11):
        folder_path = os.path.join(data_directory, f"part{i}")
        all_emails += load_emails(folder_path)
    random.shuffle(all_emails)

    for i in range(len(all_emails)):
        # Selectam un email pentru testare
        test_email = all_emails[i]
        train_emails = all_emails[:i] + all_emails[i + 1:]

        #Antrenam clasificatorul Naive Bayes fara email-ul de testare
        spam_word_counts, regular_word_counts, spam_total_words, regular_total_words, spam_email_count, regular_email_count = train_naive_bayes(
            train_emails)

        #Clasicam email-ul de testare
        predicted_label = classify_naive_bayes(test_email[0], spam_word_counts, regular_word_counts, spam_total_words,
                                               regular_total_words, spam_email_count, regular_email_count)

        if predicted_label == test_email[1]:
            accuracies.append(1)
        else:
            accuracies.append(0)

    return accuracies

accuracies = loocv(data_directory)
print("Acuratete LOOCV: ", sum(accuracies) / len(accuracies))

plt.plot(accuracies, color='blue')
plt.axhline(y=sum(accuracies) / len(accuracies), color='r', linestyle='--', label = 'Acuratete medie')
plt.xlabel('Iteratii LOOCV')
plt.ylabel('Acuratete')
plt.title('Acuratete LOOCV')
plt.legend()
plt.show()
