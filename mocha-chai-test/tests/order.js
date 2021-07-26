// based on https://gist.github.com/harto/c97d2fc9d0bfaf20706eb2acbf48c908
const chai = require('chai');
const expect = chai.expect;

describe('mocha before hooks', function () {
	before(() => console.log('*** top-level before()'));
	after(() => console.log('*** top-level after()'));
	beforeEach(() => console.log('\n*** top-level beforeEach()'));
	afterEach(() => console.log('*** top-level afterEach()'));

	describe('nesting', function () {
		before(() => console.log('  --- nested before()'));
		after(() => console.log('  --- nested after()'));
		beforeEach(() => console.log('  --- nested beforeEach()'));
		afterEach(() => console.log('  --- nested afterEach()'));

		it('is a nested spec1', () => true);
		it('is a nested spec2', () => true);
	});
});

