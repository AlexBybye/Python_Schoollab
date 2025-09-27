class Employee(object):
    # 所有Employee实例的总数
    empCount = 0

    def __init__(self, name, salary):
        """
        构造函数：初始化 Employee 对象的属性
        """
        self.name = name
        self.salary = salary

        # 每次创建新对象时，增加类变量 empCount 的值
        Employee.empCount += 1

    def displayCount(self):
        """
        实例方法：输出当前 Employee 类的实例化对象总数。
        """
        # 访问类变量时，推荐使用类名 (Employee.empCount)
        print(f"Employee 实例总数: {Employee.empCount}")

    def displayEmployee(self):
        """
        实例方法：输出当前 Employee 对象的 name 和 salary。
        """
        print(f"Name: {self.name}, Salary: {self.salary}")


# 创建Employee类的第一个对象
emp1 = Employee("Austin", 3200)
# 输出Employee类实例化的对象总数
emp1.displayCount()
# 输出Austin的信息
emp1.displayEmployee()

# 创建Employee类的第二个对象
emp2 = Employee("Tony", 5600)
# 输出Employee类实例化的对象总数
emp2.displayCount()
# 输出Tony的信息
emp2.displayEmployee()