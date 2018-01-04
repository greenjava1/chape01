#coding:utf-8
def function_with_one_star(*t):
    print(t, type(t))
def function_with_two_stars(**d):
    print(d, type(d))


function_with_one_star(1, 2, 3)
function_with_two_stars(a = 1, b = 2, c = 3)


class Dog():
    def __init__(self):
        print ("init of dog")

    def __init__(self,name):
        self.name = name
        self.age = 20
        print ("init of dog"+" name is "+name)

    def sit(self):
        print (' Dog sit')

    def roll_over(self):
        print (type(self))
        print('Dog roll_over')

class GoldenDog(Dog):
    def __init__(self):
        print ("init of GoldenDog")

    def __init__(self,name,address):
        super().__init__(name)
        self.address = address
        print ("init of GoldenDog"+" name is "+name+" address is "+address)

    def sit(self):
        print ('GoldenDog sit')

    def roll_over(self):
        print (type(self))
        print('GoldenDog roll_over')


d  =Dog('normal')
print(d.name)
print(d.age)
d1  =GoldenDog('gold1','beijing')
print(d1.name)
print(d1.age)
print(d1.address)
d.sit()
d.roll_over()

lines  = list()
with open ('记忆与印象.txt') as file_object:
    lines.append(file_object.readline())


for line in lines:
    print(line.rstrip())
print (len(lines))