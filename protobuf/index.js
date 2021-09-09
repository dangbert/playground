const fs = require('fs');
// yarn add google-protobuf

/**
 * example using protocol buffers
 */
const pBufExample = () => {
  console.log('\n\n***running protocol buf example***');
  const Schema = require('./employees_pb');
  const dan = new Schema.Employee();
  dan.setId(0);
  dan.setName('Dan');
  dan.setSalary(9001);
  console.log(`created pbuf instace with name ${dan.getName()}`);

  const bob = new Schema.Employee();
  bob.setId(1);
  bob.setName('Bob');
  bob.setSalary(2000);

  const employees = new Schema.Employees();
  employees.addEmployees(dan);
  employees.addEmployees(bob);

  // now serialize employees to binary (for sending over network etc)
  const bytes = employees.serializeBinary();
  console.log('employees data as bytes:');
  console.log(bytes);
  fs.writeFileSync('data.binary', bytes);
  console.log('wrote file data.binary');

  // deserialize:
  const employees2 = Schema.Employees.deserializeBinary(bytes);
  console.log('deserialized:');
  console.log(employees2.toString());
};

/**
 * example using json instead.
 */
const jsonExample = () => {
  console.log('\n\n***running json example***');
  const employees = []

  const dan = {
    name: 'Dan',
    salary: 9001,
    id: 0,
  }

  const bob = {
    name: 'Bob',
    salary: 2000,
    id: 1,
  }

  employees.push(dan);
  employees.push(bob);

  console.log(employees);
  fs.writeFileSync('data.json', JSON.stringify(employees));
  console.log('wrote file data.json');
};


pBufExample();
jsonExample();
