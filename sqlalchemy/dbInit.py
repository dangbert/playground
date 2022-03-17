#!/usr/bin/env python
"""demo creating sql tables in a DB given sqlalchemy models"""

import mappings
from sqlalchemy import create_engine, MetaData

DB_URI = 'sqlite:///demo.db'
engine = create_engine(DB_URI)

#meta = MetaData()
meta = mappings.mapper_registry.metadata

if input("\nDrop any existing tables? (y/n): " ).lower().strip() in ('y','yes'):
    meta.drop_all(engine)
    print("all tables dropped!")

print("\ncreating all tables")
meta.create_all(engine)
print("table list:")
print(meta.sorted_tables)

print("done!")
