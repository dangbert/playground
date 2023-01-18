% detect if we're running octave instead of matlab
isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
%fprintf('isOctave = %', isOctave)
%printf('isOctave = \n')

% print value:
isOctave

fname = 'college.csv'

if (!isOctave)
  % in matlab:
  data = readtable(fname)
else
  % in Octave you must do:
  % https://stackoverflow.com/q/32366423
  data = csvread(fname)
end

printf('read data of length = ')
length(data)
