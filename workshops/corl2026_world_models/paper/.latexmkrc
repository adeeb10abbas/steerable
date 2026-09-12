# The official style supplies its own bibliography style. Search the preserved
# template directory without copying or modifying the original distribution.
$ENV{'TEXINPUTS'} = './template//:' . ($ENV{'TEXINPUTS'} // '');
$ENV{'BSTINPUTS'} = './template//:' . ($ENV{'BSTINPUTS'} // '');
