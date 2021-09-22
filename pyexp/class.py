# This is a reference for classes in python
import datetime

class Employee:
    # pass # For temp classes

    raise_amount = 1.04 # 4% raise
    num_of_emps = 0

    def __init__(self, first, last, pay):
        self.first = first
        self.last = last
        self.pay = pay

        Employee.num_of_emps += 1

    @property # use a property decorator to signify that this method is used as an attribute
    def email(self):
        return '{}.{}@company.com'.format(self.first, self.last)

    @property
    def fullname(self):
        return '{} {}'.format(self.first, self.last)

    @fullname.setter # use a setter decorator to enable setting the attribute method
    def fullname(self, name):
        first, last = name.split(' ')
        self.first = first
        self.last = last

    @fullname.deleter # use a deleter decorator to enable setting the attribute method
    def fullname(self):
        print('Delete name!')
        self.first = None
        self.last = None

    def apply_raise(self):
        self.pay = int(self.pay * Employee.raise_amount) # or self.raise_amount

    @classmethod
    def set_raise_amount(cls, amount): # set class variables using class methods
        cls.raise_amount = amount

    @classmethod
    def from_string(cls, emp_str): # alternative constructors
        first, last, pay = emp_str.split('-')
        return cls(first, last, pay)

    @staticmethod # passing no default argument, just like functions
    def is_workday(day):
        if day.weekday() == 5 or day.weekday() == 6: # return if the day is a weekday (day # 5 or 6)
            return False
        return True

    def __repr__(self): # special method in class useful for representing information about objects
        return "Employee('{}', '{}', {})".format(self.first, self.last, self.pay)

    def __str__(self, other): # a more user-friendly version, used in higher priority
        return NotImplemented # see if the other object knows how to handle "str" method

    def __len__(self, arg): # also a length special methods
        pass

class Developer(Employee): # inheriting from "Employee" class
    """docstring for Developer."""
    def __init__(self, first, last, pay, prog_lang):
        super(Developer, self).__init__(first, last, pay) # Let the superclass to handle initialization of the old arguments
        self.prog_lang = prog_lang

class Manager(Employee):
    """docstring for Manager."""
    def __init__(self, first, last, pay, employees = None):
        super(Manager, self).__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emp(self):
        for emp in self.employees:
            print('-->', Employee.fullname())

# %%
emp_1 = Employee('Hello', 'World', 6000) # initialize a emp class
print(emp_1.email)
print(emp_1.fullname)

emp_1.apply_raise()
print(emp_1.pay)
print(Employee.__dict__) # print the namespace for the class
# we can also set the class variable manually in the code
print(Employee.num_of_emps)

Employee.set_raise_amount(1.05)

print(Employee.raise_amount)

# %%
emp_str_2 = 'John-Doe-7000'
emp_2 = Employee.from_string(emp_str_2)
print(emp_2.fullname)
del emp_2.fullname # this triggers the deleter method

# %%
date = datetime.date(2017, 5, 6)
print(Employee.is_workday(date))

# %%
print(isinstance(emp_1, Employee)) # see if something is an instance
# there are also 'issubclass()', etc.

# %%
# the following section is about error and exception handling
try:
    pass # some possible bad codes
    raise Exception # can raise exceptions manually
except FileNotFoundError as err: # specific error handling
    print(err) # can print default error message
except Exception: # most general exception handling, will run the code below in every exception
    print('Sorry, something went wrong.')
else: # optional
    pass # execute the code
finally: # runs no matter what happens in the code (success or not)
    pass
# end of section
