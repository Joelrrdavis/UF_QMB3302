"""

Python Foundations: Data, Control, and Functions

This file is intentionally more detailed than the textbook.

Run it top-to-bottom. Then modify small pieces and observe what changes. 

Some of this material (a lot?) overlaps with prior and future content. Intentionally. 
"""

# ------------------------------------------------------------
# 1. Python Basics: Variables and Data Types
# ------------------------------------------------------------

2+2
print(2+2)

x = 2
print(x+2)

# Uncomment to see the error:
#print(y)
y = 3
print(x+y)

#what's with the below? It's just a section break I am printing,
# in case you want to scroll back to find something particular. 
print("\n=== 1. Variables and Data Types ===")

# 1.1 Variables revisited (reassignment)
x = 2
print("x starts as:", x)

# Variables are names pointing to values. The name can be reassigned.

x = 10
print("x after reassignment:", x)

x = 3
print (x)

# 1.2 Numbers and numeric behavior
a = 10
b = 3

print("\n-- Numbers --")
print("a =", a)
print("b =", b)

print("a / b =", a / b)       # division produces a float
print("a // b =", a // b)     # floor division
print("a % b =", a % b)       # remainder (modulo)
print("a ** b =", a ** b)     # exponentiation

# Floating-point representation surprise (not a bug, just how floats work)
print("\nFloating-point example:")
print("0.1 + 0.2 =", 0.1 + 0.2)
print("(0.1 + 0.2) == 0.3 ?", (0.1 + 0.2) == 0.3)


# 1.3 Strings as data
print("\n-- Strings --")
text = "analytics"
print("text =", text)
print("First character text[0] =", text[0])
print("Last character text[-1] =", text[-1])

# Strings are immutable (cannot change characters in place)
# Uncomment to see the error:
#text[0] = "A"  # TypeError


# 1.4 Booleans and logical values
print("\n-- Booleans --")
x = 5
print("x =", x)
print("x > 3:", x > 3)
print("x == 5:", x == 5)
print("x > 10:", x > 10)


# 1.5 Type behavior and common surprises
print("\n-- Types and surprises --")
print('"5" + "5" =', "5" + "5")  # concatenation
print("5 + 5 =", 5 + 5)          # addition

# Mixing types can raise errors
# Uncomment to see the error:
# print("5" + 5)  # TypeError

# Fixes by explicit conversion (use deliberately)
print('"5" + str(5) =', "5" + str(5))
print("int('5') + 5 =", int("5") + 5)

print("\nTypes with type():")
print("type(5) =", type(5))
print("type(5.0) =", type(5.0))
print("type('five') =", type("five"))
print("type(True) =", type(True))



# ------------------------------------------------------------
# 2. Working with Strings
# ------------------------------------------------------------

print("\n=== 2. Working with Strings ===")

# 2.1 Creation and indexing
phrase = "Warrington College"
print("phrase =", phrase)
print("phrase[0] =", phrase[0])
print("phrase[1] =", phrase[1])
print("phrase[-1] =", phrase[-1])

# Indexing out of range causes an error
# Uncomment to see the error:
#print(phrase[999])  # IndexError

# 2.2 Slicing (start inclusive, end exclusive)
word = "warrington"
print("\nSlicing examples with:", word)
print("word[0:4] =", word[0:4])
print("word[:4] =", word[:4])
print("word[4:] =", word[4:])
print("word[-3:] =", word[-3:])

# 2.3 Common methods (strings are immutable; methods return new strings)
print("\nString methods:")
print("len(phrase) =", len(phrase))
print("phrase.lower() =", phrase.lower())
print("phrase.upper() =", phrase.upper())
print("phrase.replace('Warrington','College') =", phrase.replace("Warrington", "College"))

original = "Hello"
modified = original.upper()
print("\nImmutability check:")
print("original =", original)
print("modified =", modified)

# 2.4 Formatting strings (f-strings)
name = "Joel"
score = 92
points_possible = 100
print("\nF-string formatting:")
print(f"{name} scored {score} points")
print(f"{name} earned {score}/{points_possible} = {score/points_possible:.2%}")
# this is kind of hard... the best way to understand these is just try a bunch of them. 


# ------------------------------------------------------------
# 3. Data Structures: Lists
# ------------------------------------------------------------

print("\n=== 3. Lists ===")

# 3.1 What a list is
numbers = [1, 2, 3, 4]
print("numbers =", numbers)

# 3.2 Creating lists
names = ["Alice", "Bob", "Charlie"]
mixed = [1, "two", True]
print("names =", names)
print("mixed =", mixed)

results = []
print("results (empty list) =", results)

# 3.3 Indexing and slicing lists
values = [10, 20, 30, 40, 50]
print("\nIndexing and slicing:")
print("values =", values)
print("values[0] =", values[0])
print("values[-1] =", values[-1])
print("values[1:4] =", values[1:4])

# IndexError example
# Uncomment to see the error:
#print(values[99])  # IndexError

# 3.4 Modifying list contents (lists are mutable)
values = [10, 20, 30]
print("\nMutating a list:")
print("values (before) =", values)

values[1] = 25
print("values (after values[1]=25) =", values)

values.append(40)
print("values (after append(40)) =", values)

values.remove(25)
print("values (after remove(25)) =", values)

popped = values.pop(0)
print("popped value =", popped)
print("values (after pop(0)) =", values)

# 3.5 Common patterns and mistakes
print("\nBuilding a list with append:")
built = []
for n in [1, 2, 3]:
    built.append(n * 10)
print("built =", built)

print("\nReference surprise (two names, one list):")
a = [1, 2, 3]
b = a
b.append(4)
print("a =", a)
print("b =", b)

print("\nCopying to avoid shared references:")
a = [1, 2, 3]
b = a.copy()
b.append(4)
print("a =", a)
print("b =", b)


# ------------------------------------------------------------
# 4. Data Structures: Dictionaries
# ------------------------------------------------------------

print("\n=== 4. Dictionaries ===")

person = {"name": "Alex", "age": 30}
print("person =", person)

scores = {"math": 90, "history": 85}
print("scores =", scores)
print("scores['math'] =", scores["math"])

# KeyError example
# Uncomment to see the error:
#print(scores["science"])  # KeyError

print("\nSafer access with get():")
print("scores.get('science') =", scores.get("science"))
print("scores.get('science', 0) =", scores.get("science", 0))

print("\nUpdating dictionaries:")
scores["math"] = 92
scores["science"] = 88
print("scores (updated) =", scores)

print("\nIterating over keys:")
for subject in scores:
    print(" -", subject)

print("\nIterating over key-value pairs:")
for subject, value in scores.items():
    print(f" - {subject}: {value}")

print("\nList of dictionaries (structured records):")
students = [
    {"name": "Alex", "score": 90},
    {"name": "Jordan", "score": 85},
    {"name": "Casey", "score": 93},
]
print("students =", students)


# ------------------------------------------------------------
# 5. Control Flow: Conditional Logic
# ------------------------------------------------------------

# A conditional lets Python make a decision.
# The simplest possible conditional compares two values.

# A simple comparison


# This condition is either True or False.
if 110 < 120:
    print("110 is less than 120")

# Using a variable in a condition

score = 85

# Here, Python checks whether the condition is True.
# Since score = 85, this block will NOT run.
if score >= 90:
    print("Score is at least 90")





# Adding an alternative with else

score = 85

if score >= 90:
    grade = "A"
else:
    grade = "Not an A"

print("score =", score, "-> grade =", grade)



# Multiple conditions with elif

# elif means: "otherwise, if this condition is True"
# Try changing score to 95, 80, or 70 and re-run.
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "Below B"

print("score =", score, "-> grade =", grade)



# Conditions using ranges (and)

age = 20

# The word "and" means BOTH conditions must be True.
if age >= 18 and age < 21:
    print("Adult, but under 21")

# This will run because:
# - age >= 18 is True
# - age < 21 is also True



# Adding elif to handle multiple age ranges
# Only ONE of these blocks will run.

age = 20

if age < 18:
    print("Minor")
elif age >= 18 and age < 21:
    print("Adult, but under 21")
else:
    print("Adult, 21+")


# Nested conditionals (same logic, written step-by-step)

age = 20

if age >= 18:
    # We only get here if the first condition is True
    if age < 21:
        print("Nested: adult, but under 21")
    else:
        print("Nested: adult, 21+")
else:
    print("Nested: minor")

# This nested version is equivalent to the previous example,
# but written in smaller decision steps.

# Key ideas to remember
# - Conditions must evaluate to True or False
# - Indentation defines what code belongs to each condition
# - elif lets you check multiple possibilities
# - Only one branch of an if / elif / else chain runs
# - Nested conditionals break complex decisions into smaller ones

# ------------------------------------------------------------
# 6. Control Flow: Loops
# ------------------------------------------------------------

# For loop over a list
values = [10, 20, 30]
for v in values:
    print("Value:", v)

# For loop over a string
for ch in "AI":
    print("Character:", ch)

# Accumulation pattern
numbers = [1, 2, 3, 4]
running_sum = 0
for n in numbers:
    running_sum += n
print("\nManual sum:", running_sum, "(compare to sum(numbers) =", sum(numbers), ")")

# While loop with termination
count = 0
while count < 3:
    print("count =", count)
    count = count + 1

# Infinite loop example (commented out)
# count = 0
# while count < 3:
#     print(count)
#     # count never changes -> infinite loop

# Off-by-one demonstration
values = [10, 20, 30]
for i in range(len(values)):
    print(f"Index {i} -> {values[i]}")

# Uncomment to see an IndexError:
# for i in range(len(values) + 1):
#     print(values[i])

# Modifying a list while iterating can be surprising
nums = [1, 2, 3, 4, 5, 6]
#Removing items while iterating (demonstration):
nums_copy = nums.copy()
for n in nums_copy:
    if n % 2 == 0:
        nums_copy.remove(n)
print("Result:", nums_copy)

# Safer pattern: build a new list instead:
nums = [1, 2, 3, 4, 5, 6]
odds = []
for n in nums:
    if n % 2 != 0:
        odds.append(n)
print("odds =", odds)


# ------------------------------------------------------------
# 7. Functions and Modular Design
# ------------------------------------------------------------

print("\n=== 7. Functions ===")

def greet():
    print("Hello")

print("\nCalling greet() twice:")
greet()
greet()

def add(a, b):
    return a + b

result = add(2, 3)
print("\nadd(2, 3) returns:", result)

def add_and_print(a, b):
    print("Inside function:", a + b)
    # no return statement -> returns None implicitly

returned_value = add_and_print(2, 3)
print("add_and_print(2, 3) returns:", returned_value)

def is_passing(score):
    return score >= 70

print("\nUsing is_passing(score):")
for s in [65, 70, 85]:
    print("score =", s, "passing?", is_passing(s))

def average_score(records):
    """
    records: list of dictionaries, each with a 'score' key
    returns: float average score
    """
    total = 0
    for rec in records:
        total += rec["score"]
    return total / len(records)

students = [
    {"name": "Alex", "score": 90},
    {"name": "Jordan", "score": 85},
    {"name": "Casey", "score": 93},
]
print("\nAverage score (via function):", average_score(students))


# ------------------------------------------------------------
# 8. Debugging Fundamentals
# ------------------------------------------------------------


# 8.1 Syntax errors stop the program before it runs
# Uncomment to see a SyntaxError:
# print("Hello"   # missing closing parenthesis

# 8.2 Tracebacks show where execution failed
def divide(a, b):
    return a / b

# Uncomment to see a ZeroDivisionError traceback:
# divide(10, 0)

# 8.3 Common beginner errors (commented out)

# NameError: variable not defined
# print(not_defined_yet)

# TypeError: incompatible types
# print("5" + 5)

# IndexError: out of range
#vals = [1, 2, 3]
#print(vals[3])

# KeyError: missing key
#d = {"a": 1}
#print(d["b"])

# 8.4 Systematic debugging example: inspect assumptions
scores = [90, 85, 93]
avg = sum(scores) / len(scores)
print("\nAverage score computed as sum(scores)/len(scores):", avg)
print("sum(scores) =", sum(scores))
print("len(scores) =", len(scores))

# 8.5 Logic vs code bug example
def is_valid(score):
    # This runs fine, but the logic might not match our intent.
    return score > 100

print("\nLogic check for is_valid(score) where intent is unclear:")
for s in [50, 75, 110]:
    print("score =", s, "-> is_valid?", is_valid(s), "(Does this match your intended rule?)")

