The short answer is that you should dev MyPkg rather than add MyPkg.

In greater detail: when you say ]add MyPkg,the package manager makes a copy of the files that you created in D:\work\julia\test and stashes them in your .julia/packages directory. Right after the add you’ll probably get something like this:

julia> pathof(MyPkg)
"/home/tim/.julia/packages/MyPkg/o1AMF/src/MyPkg.jl"

This is the path that Revise uses as the definition of this package. Consequently any changes to files in D:\work\julia\test do not get tracked.

Even if you edited the files in .julia/packages/MyPkg/, Revise wouldn’t track them. That’s because add means “use a version-controlled release of a given package.” If you’re making changes, you’re no longer using a version-controlled release. Indeed, in recent Julia versions all the source files in .julia/packages have write permissions disabled. (Your editor might “helpfully” offer to change them to writable, but you should decline such offers of assistance.)

Revise will, however, follow changes from add to dev. If you make a mistake and ]add MyPkg instead, you can fix it by saying ]dev MyPkg and any changes you’ve already made will be incorporated. That’s because Revise will use the files in your .julia/packages/MyPkg directory as a reference against which to compare the changes.
