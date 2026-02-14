"""
week01_main.py

This section that start with 3 " above is intended to be commentary, 
it does "do" anything so it is not run as code.

Comments are also indicated with a single #, you'll see them below. 

Week 1: Printing, Variables, and Simple Math

"""

# --------------------------------------------------
# 7.1 Printing Output
# --------------------------------------------------

# Basic printing
print("Hello, world!")
print(3)
print(2 + 5)

# Printing multiple values
price = 20
tax = 1.50
print("Price:", price, "Tax:", tax)

# Controlling print behavior
print("A", "B", "C", sep=" | ")   # custom separator
print("End of line example", end=" <-- still same line\n")

# Debug-style printing (very common pattern)
print("DEBUG: price =", price)
print("DEBUG: tax =", tax)

# --------------------------------------------------
# 7.2 Variables and Assignment
# --------------------------------------------------

# Variable assignment
x = 2
print("x =", x)

# Reassignment (variables can change!)
x = x + 1
print("x after reassignment =", x)

# Variables as labels for values
quantity = 3
total_items = quantity * x
print("Total items:", total_items)

# Variable naming conventions (snake_case)
unit_price = 5.00
total_cost = unit_price * quantity
print("Total cost:", total_cost)

# Types 
print("Type of unit_price:", type(unit_price))
print("Type of quantity:", type(quantity))
print("Type of 'Hello':", type("Hello"))

# --------------------------------------------------
# 7.3 Simple Math and Expressions
# --------------------------------------------------

a = 10
b = 4

print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)

# Integer vs float division
print("9 / 2 =", 9 / 2)     # float division
print("9 // 2 =", 9 // 2)   # floor division
print("9 % 2 =", 9 % 2)     # remainder (modulo)

# Exponentiation
print("2 ** 3 =", 2 ** 3)

# Order of operations
print("2 + 3 * 4 =", 2 + 3 * 4)
print("(2 + 3) * 4 =", (2 + 3) * 4)

# --------------------------------------------------
# 7.4 A First Complete Python Program
# --------------------------------------------------

def compute_total(price, tax):
    """
    Compute the total cost given a price and tax.
    """
    return price + tax


def main():
    price = 20
    tax = 1.50

    total = compute_total(price, tax)

    print("Price:", price)
    print("Tax:", tax)
    print("Total cost:", total)


# This ensures the program runs only when this file is executed directly
if __name__ == "__main__":
    main()

# --------------------------------------------------
# Common Beginner Mistakes (Commented Out)
# --------------------------------------------------

# 1. Missing quotes around text
# print(Hello)               # NameError
# print("Hello")             # Correct

# 2. Using a variable before assignment
# print(total)               # NameError
# total = 10
# print(total)

# 3. Case sensitivity
# Price = 20
# print(price)               # NameError (price != Price)

# 4. Confusing assignment with equality
# x == 5                     # Does nothing
# x = 5                      # Correct assignment
