List Operations
# returns the length of a list:len(my_collection)
# Add multiple items to a list:my_collection.extend([ More", "Items"])
# Add a single item to a list:my_collection.append("Single")
# Delete the object of a list at a specified index:del(my_collection[2])
# Clone a list:clone = my_collection[:]
# Concatenate two lists:my_collection_3 = my_collection + my_collection_2
# Calculate the sum of a list of ints or floats:sum(number_collection)
# Check if an item is in a list, returns Boolean:item (not) in my_collection

Set:Unordered collection of unique objects
Set Operations
# Convert a list to a set:my_set = set([1,1,2,3])
# Add an item to the set:a.add(4)
# Remove an item from a set:a.remove("Bye")
# Returns set a minus b:a.difference(b)#a-b
# Returns intersection(交集) of set a and b:a.intersection(b)
# Returns the union of set a and b:a.union(b)
# Returns True if a is a subset/superset of b, false otherwise:a.issubset/issuperset(b)

if statement_1:
elif statement_2:
else:
range(start, stop, step)

# Import BeautifulSoup
from bs4 import BeautifulSoup
# Parse HTML stored as a string
soup = BeautifulSoup(html, 'html5lib')
# Returns formatted html
soup.prettify()
# Find the first instance of an HTML tag
soup.find(tag)
# Find all instances of an HTML tag
soup.find_all(tag)

# Import the requests library

import requests
# Send a get requests to the url with optional parameters
response = requests.get(url, parameters)
# Get the url of the response
response.url
# Get the status code of the response
response.status_code
# Get the headers of the request
response.request.headers
# Get the body of the requests
response.request.body
# Get the headers of the response
response.headers
# Get the content of the response in text
response.text
# Get the content of the response in json
response.json()
# Send a post requests to the url with optional parameters
requests.post(url, parameters)

# Create a function
def function_name(optional_parameter_1, optional_prameter_2):
 # code to execute
 return optional_output
# Calling a function
output = function_name(parameter_1, parameter_2)

#读写文件
exmp2 = '/Example2.txt'
with open(exmp2, 'w') as writefile:
    writefile.write("This is line A")

with open(exmp2, 'r') as testwritefile:
    print(testwritefile.read())

with open('/Example2.txt', 'a') as testwritefile:#不丢失现有数据
    testwritefile.write("This is line C\n")
    testwritefile.write("This is line D\n")
    testwritefile.write("This is line E\n")

r+：读取和写入。无法截断文件。.truncate()删除指针后面的数据
w+：读取和写入。截断文件。覆盖文件
a+ ：追加和读取。如果不存在，则创建一个新文件。您不必详细了解本实验的每种模式的具体细节。使用a+打开文件后光标停留在最后
  .tell() - 返回当前位置（以字节为单位）
  .seek(offset,from) - 将指针位置相对于“from”更改“offset”字节。 from可以取0,1,2对应的值开始，相对于当前位置和结束
     .seek(0,0) # move 0 bytes from beginning.

# Opens a file in read mode
file = open(file_name, r")
# Returns the file name
file.name
# Returns the mode the file was opened in
file.mode
# Reads the contents of a file
file.read()
# Reads a certain number of characters of a file
file.read(characters)
# Read a single line of a file
file.readline()
# Read all the lines of a file and stores it in a list
file.readlines()
# Closes a file
file.close()

Writing to a File
# Opens a file in write mode
file = open(file_name, w )
# Writes content to a file
file.write(content)
# Adds content to the end of a file
file.append(content)

Objects and Classes
# Creating a class
class class_name:
 def __init__(self. optional_parameter_1, optional_parameter_2):
 self.attribute_1 = optional_parameter_1
 self.attribute_2 = optional_parameter_2
 def method_name(self, optional_parameter_1):
 # Code to execute
 return optional_output
# Create an instance of a class
object = class_name(parameter_1, parameter_2)
# Calling an object method
object.method_name(parameter_3)
            
#复制文件
with open('/Example2.txt','r') as readfile:
    with open('Example3.txt','w') as writefile:
          for line in readfile:
                writefile.write(line)

#定义类Circle和方法
import matplotlib.pyplot as plt
%matplotlib inline  
class Circle(object):
    
    # Constructor
    def __init__(self, radius=3, color='blue'):
        self.radius = radius
        self.color = color 
    
    # Method
    def add_radius(self, r):
        self.radius = self.radius + r
        return(self.radius)
    
    # Method
    def drawCircle(self):
        plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        plt.axis('scaled')
        plt.show()  
      
circle1=Circle(10, 'red')
dir(circle1)
#错误处理
a = 1
try:b = int(input("Please enter a number to divide a"))
    a = a/b
except ZeroDivisionError:
    print("The number you provided cant divide 1 because it is 0")
except ValueError:
    print("You did not provide a number")
except:
    print("Something went wrong")
else:
    print("success a=",a)
finally:
    print("Processing Complete")

