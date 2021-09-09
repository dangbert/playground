const fs = require('fs');


const employees = []

const dan = {
  name: 'Dan',
  salary: 1000,
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
