#write a function that has 
#	one non-default parameter
#	one default parameter
#	handles extra command parameters
#	handles extra named parameters
# and prints each parameter as name = value on separate lines. Unnamed parameters should have "unnamed1", "unnamed2", etc. for their names

def morning(name, msg = "Good Morning", *friends, **mood):
	print("Hello",  name, msg)
	if name = null:
		return unnamed1

	for friend in friends:
		print("And hello " + friend)
		if friend = null; 
			return unnamed2

	for key, value in mood.items() :
		print(key, "=", value)



print(morning("Kate"))
print(morning("Lucy", "how are you" , "Marcus", "Sandy", marcus = 1, sandy = 5))
print(morning())


