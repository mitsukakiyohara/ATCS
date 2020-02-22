#Write a function that asks the user for a cost in pennies, and does the right thing whether is a string, integer or a decimal 
#print error message and ask again (str)
#return value(int)
#assume it's in dollars, convert in pennies (in integer) and return 


def askAboutPennies():
	try:
		userInput = input("enter a cost in pennies: ")
		output = int(userInput)
		return output
	except ValueError as verror: 
		try: 
			output = float(userInput)
			output = output * 100
			return int(output)
		except ValueError as verror:
			print("this is a string. try again.")
			return askAboutPennies()

print(askAboutPennies())
	

	
	




	





