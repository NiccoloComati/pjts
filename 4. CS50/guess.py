def get_guess():
    guess = int(input("Enter your guess: "))
    return guess

def main():
    guess = get_guess()
    if guess == 42:
        print("You guessed correctly!")
    else:
        print("You guessed incorrectly!")


main()