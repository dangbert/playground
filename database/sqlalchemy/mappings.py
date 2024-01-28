#!/usr/bin/env python
from sqlalchemy.orm import registry, declarative_base, declared_attr
from sqlalchemy import Table, Column, Integer, String

# registry maintains a collections of mappints
#   and provides configuration hooks for mapping classes
mapper_registry = registry()


# 3 kinds of mapping (can be used interchangebly)
#   https://docs.sqlalchemy.org/en/14/orm/mapping_styles.html

#### 1. Declarative Mapping
#   ***"the typical way mappings are constructed in modern SQLAlchemy"***
#   (tldr use this method)
#   "the Base class refers to a registry object that maintains a collection of related mapped classes"
#    https://docs.sqlalchemy.org/en/14/orm/mapping_styles.html#declarative-mapping

# this is supposed to function the same as below, BUT it doesn't (customer table doesn't get created)
#Base = declarative_base()
Base = mapper_registry.generate_base() # the above is (supposedly) shorthand for this

# example mapping using the base class:
class Customer(Base):
    __tablename__ = "customer"
    id = Column(Integer, primary_key=True, autoincrement = True)
    name = Column(String)


#### 2. Declarative Decorator
#registry.mapped()

# decorator invokes registry.generate_base() for Base class:
#  (passing through any args provided to the decorator)
@mapper_registry.as_declarative_base()
class Base2(object):
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()
    id = Column(Integer, primary_key=True)

# now we can use our base class:
class Products(Base2):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True, autoincrement = True)
    name = Column(String)


#### 3. Imperative (a.k.a. Classical) Mappings:
#  (where the target class doesn't include aany declarative class attributes)
#  https://docs.sqlalchemy.org/en/14/orm/mapping_styles.html#imperative-a-k-a-classical-mappings

dog_table = Table(
    "dog", # table name
    mapper_registry.metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String(50))
)
class Dog:
    pass
mapper_registry.map_imperatively(Dog, dog_table)
# you can also define relationships to other classes etc (see docs link above)


##### General Info:
#   mapped class behavior:
#   https://docs.sqlalchemy.org/en/14/orm/mapping_styles.html#mapped-class-behavior
