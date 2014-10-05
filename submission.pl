#!/usr/bin/perl
use File::Compare;
use strict;

print "Checking for input.h and main.cu ... ";
my $fileName = "input_large.h";
unless(-e $fileName) {
    print "\n[Error] File $fileName does not exist.\n";
    exit 1;
}

my $fileName = "main.cu";
unless(-e $fileName) {
    print "\n[Error] File $fileName does not exist.\n";
    exit 1;
}
print "done\n";

print "Checking for CUDA environment ... ";
# Compile the code
system("which nvcc > /dev/null 2>&1");
if($?) {
    print "\n[Error] The PATH environment variable does not include the toolkit bin directory.\n";
    print "[Error] Make sure the PATH and LD_LIBRARY_PATH environment variables are properly set.\n";
    print "[Error] For more information see http://docs.nvidia.com/cuda/index.html#getting-started-guides \n";
    exit $? >> 8;
}
print "done\n";

print "Checking for compilation ... ";
system("nvcc -arch sm_35 -o hillsAndDales main.cu > /dev/null 2>&1");
if($?) {
    system("nvcc -arch sm_35 -o hillsAndDales main.cu > /dev/null 2>&1");
    print "\n[Error] Compilation failed.\n";
    print "\n[Error] Resolve any compilation errors and try again.\n";
    exit $? >> 8;
}
print "done\n";

print "Checking for execution correctness ... ";
# Run the code
system("./hillsAndDales > hillsAndDales.log");
if($?) {
    print "\n[Error] Ensure that the application returns 0 exit code in case of successful execution.\n";
    exit $? >> 8;
}
print "done\n";

print "Checking for application output ... ";
# Compare the file with expected output
if(compare("hillsAndDales.log","expected_output_large")) {
    print "\n[Error] Application output does not match with the expected output.\n";
    print "\n[Error] Application Output\n";
    if(open(MYFILE, "hillsAndDales.log")) {
        my $line = <MYFILE>;
        while($line ne "") {
            print $line;
            $line = <MYFILE>;
        }
    }
    close MYFILE;
    
    print "[Error] Expected Output\n";
    if(open(MYFILE, "expected_output_large")) {
        my $line = <MYFILE>;
        while($line ne "") {
            print $line;
            $line = <MYFILE>;
        }
    }
    close MYFILE;

    exit 1;
}
print "done\n";

system("time -f \"real %es\nuser %Us\nsystem %Ss\" ./hillsAndDales");
print "All done, SUCCESS!\n";
exit 0;