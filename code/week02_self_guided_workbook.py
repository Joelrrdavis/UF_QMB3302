"""
week02_self_guided_workbook.py

Week 2 — Self-Guided Practice Workbook 

This workbook doesn't have a separate "solutions" guide. It should
be obvious by following the code when you have done something
right/wrong. 

Focus:
- Variables and basic types (int, float, str, bool)
- String operations (concatenation, f-strings, .split, .strip, .lower)
- Lists (indexing, slicing, append, len, sum)
- Loops (for) and simple accumulation patterns
- Conditionals (if / elif / else) and compound conditions
- Dictionaries (key/value) for counting categories
- A small "mini-lab" at the end that combines everything

How you know you’re “right”:
- You are NOT trying to match an exact number (mostly true... read the notes).
- You are looking for patterns described in the CHECK YOURSELF notes.
- If your outputs match those patterns, you are right on track.

------------------------------------------------------------
You can run this file a few ways... to run the whole file in the terminal:

- In VS Code terminal:
    python week02_self_guided_workbook.py

On some machines you may need to try python3 instead of python. 
You can also run it line by line as you see me doing in the course videos. 
------------------------------------------------------------
"""

from __future__ import annotations


# ============================================================
# 0) Warm-up: variables + types
# ============================================================

# Variables store values. Python figures out the type automatically.
x = 10
y = 2.5
name = "Avery"
is_student = True

print("\n[0] Types of a few variables:")
print(type(x), type(y), type(name), type(is_student))

# TRY THIS:
# - Change x to a decimal (e.g., 10.0). What does its type become?
# - Change is_student to False. What stays the same?
#
# CHECK YOURSELF:
# - int vs float is determined by whether there is a decimal point.
# - bool is either True or False.


# ============================================================
# 1) Arithmetic and assignment
# ============================================================

a = 7
b = 3

#Basic arithmetic:
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)   # division produces a float
print("a // b =", a // b) # floor division
print("a % b =", a % b)   # remainder (mod)

# TRY THIS:
# - Set b = 2 and rerun. What happens to // and %?
#
# CHECK YOURSELF:
# - a = (a // b) * b + (a % b) always holds for integers.


# ============================================================
# 2) Strings: building text and extracting pieces
# ============================================================

first = "Sam"
last = "Lee"
full = first + " " + last

print("\n[2] String building:")
print(full)

# f-strings are the cleanest way to mix text + variables
age = 19
print(f"{first} {last} is {age} years old.")

# Useful cleaning operations
# Remember you don't need to memorize syntax like .strip. 
raw = "  Golden Retriever  "
clean = raw.strip().lower()

print("\n[2] String cleaning:")
print("raw:", repr(raw))
print("clean:", repr(clean))

# Splitting text into parts (common in data work)
text = "A,B,C"
parts = text.split(",")
print("\n[2] Splitting:")
print(parts)

# TRY THIS:
# - Change text to "A, B, C" (notice spaces). Use strip() on each part.
# - Use .upper() instead of .lower().
#
# CHECK YOURSELF:
# - .strip() removes whitespace at the edges.
# - .split(",") returns a list.


# ============================================================
# 3) Lists: indexing, slicing, and basic summaries
# ============================================================

scores = [90, 85, 93, 72, 88]

# List basics:
print("scores:", scores)
print("first score:", scores[0])
print("last score:", scores[-1])
print("first 3 scores:", scores[0:3])
print("number of scores:", len(scores))

# Simple summaries:
print("min:", min(scores))
print("max:", max(scores))
print("sum:", sum(scores))
print("mean (sum/len):", sum(scores) / len(scores))

# Adding values
scores.append(95)
#After append(95):
print(scores)

# TRY THIS:
# - Append a few more scores.
# - Remove one score by value using scores.remove(72) (if it exists).
#
# CHECK YOURSELF:
# - len(scores) changes when you append/remove.
# - mean changes if you add unusually high or low values.


# ============================================================
# 4) Loops: doing something to every item (they use iterable lists etc)
# ============================================================

# Loop through scores and classify them:

for s in scores:
    if s >= 90:
        print(s, "-> A-range")
    elif s >= 80:
        print(s, "-> B-range")
    else:
        print(s, "-> below B-range")

# TRY THIS:
# - Change the cutoffs (e.g., A >= 93). How do classifications change?
#
# CHECK YOURSELF:
# - Each score is classified once.
# - Only one branch (if/elif/else) runs per score.


# ============================================================
# 5) Accumulation patterns: counting and averaging “by hand”
# ============================================================

# Counting how many scores are >= 80
count_80_plus = 0
for s in scores:
    if s >= 80:
        count_80_plus += 1

print("\n[5] Count of scores >= 80:", count_80_plus)

# Compute an average of only scores >= 80 (filter + accumulate)
total_80_plus = 0
count_80_plus = 0

for s in scores:
    if s >= 80:
        total_80_plus += s
        count_80_plus += 1

if count_80_plus > 0:
    avg_80_plus = total_80_plus / count_80_plus
    print("[5] Average of scores >= 80:", avg_80_plus)

# TRY THIS:
# - Change the threshold to 90 and rerun.
#
# CHECK YOURSELF:
# - As the threshold increases, count typically decreases.
# - The average of the remaining scores typically increases.


# ============================================================
# 6) Compound conditions (and / or) — keep it simple
# ============================================================

age = 20

#Compound condition example:
if age >= 18 and age < 21:
    print("Adult, but under 21")
elif age >= 21:
    print("Adult, 21+")
else:
    print("Minor")

# TRY THIS:
# - Set age = 17, then 21, then 35 and rerun.
#
# CHECK YOURSELF:
# - Exactly one message prints each time.


# ============================================================
# 7) Dictionaries: counting categories
# ============================================================

# A dictionary maps keys -> values.
# Here we count how many times each letter appears.
letters = ["A", "B", "A", "C", "B", "A", "C", "A"]

counts = {}  # empty dictionary

for letter in letters:
    if letter in counts:
        counts[letter] += 1
    else:
        counts[letter] = 1

# Counts dictionary:
print(counts)

# TRY THIS:
# - Add more letters to the list (including a new letter like "D").
# - Observe how the dictionary changes.
#
# CHECK YOURSELF:
# - Keys are unique categories.
# - Values are the counts for each category.


# ============================================================
# 8) Really, really optional. 
# Mini-lab: simple “student records” analysis 
# ============================================================
#
# This is far more than you need to know.
# 
# You have a tiny dataset: each record is a string.
# Your job: clean it, parse it, summarize it.


records = [
    "Alex, Finance, 90",
    " Jordan , Marketing , 85 ",
    "Casey, IS, 93",
    "Riley, Finance, 72",
    "Morgan, IS, 88",
]

# Raw records (first 2):
print(records[0])
print(records[1])

# Step 1: parse into a cleaner structure
# We will build three lists: students, majors, scores_parsed
students = []
majors = []
scores_parsed = []

for r in records:
    parts = r.split(",")

    # Clean spaces around text
    student = parts[0].strip()
    major = parts[1].strip()
    score = int(parts[2].strip())  # convert score to integer

    students.append(student)
    majors.append(major)
    scores_parsed.append(score)

print("\n[8] Parsed lists:")
print("students:", students)
print("majors:", majors)
print("scores:", scores_parsed)

# Step 2: overall mean score (one number)
overall_mean = sum(scores_parsed) / len(scores_parsed)
print("\n[8] Overall mean score:", overall_mean)

# Step 3: count students per major (dictionary)
major_counts = {}
for m in majors:
    if m in major_counts:
        major_counts[m] += 1
    else:
        major_counts[m] = 1

print("\n[8] Students per major:")
print(major_counts)

# Step 4: mean score by major (dictionary accumulation)
major_totals = {}
major_n = {}

for m, s in zip(majors, scores_parsed):
    major_totals[m] = major_totals.get(m, 0) + s
    major_n[m] = major_n.get(m, 0) + 1

major_means = {m: major_totals[m] / major_n[m] for m in major_totals}

print("\n[8] Mean score by major:")
print(major_means)

# TRY THIS:
# 1) Add a new record with extra spaces (e.g., "Taylor , Finance , 91")
# 2) Add a new major (e.g., "Accounting")
# 3) Add a record with a missing score and decide how you want to handle it
#    (skip it, or use 0, or print a warning).
#
# CHECK YOURSELF:
# - strip() should remove extra spaces consistently.
# - counts and means should update when you add records.
# - handling missing data forces you to define a rule (this is exactly what pandas will help with).

