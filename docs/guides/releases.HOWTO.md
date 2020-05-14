# Release HOWTO

Here the instructions for create a new package distribution Release.

## Release of a new major/minor version

Create a new branch called releases/<X>.<Y>.x where <X>.<Y> is the corresponding release version. 
For example, for a new release 1.3, you must create a branch called `releases/1.3.x

```shell
git checkout -b releases/1.3.x
```


Push the new branch
```shell
git push origin releases/1.3.x
```

Check all release stuff (version, documentation, pypi-test, tests,...)


If everything is ok, you can create the specific version tag

```shell
git tag -a 1.3.0 -m "version 1.3.0"
git push origin 1.3.0
```

This action will trigger a ci pipeline where all related resources will be deployed


## Release of a patch version

If you need apply a hotfix or a bugfix into an release. You must follow these steps:

Checkout the release branch you want to fix (in our case 1.3.x):

```shell
git checkout releases/1.3.x
```

Create a bugfix/hotfix branch from here:

```shell
git checkout -b bugfix/#<id_of_related_issue>
```

Follow common dev process (PR + approvals ...) and squash fix into release branch, then check all 
release stuff (version, documentation, pypi-test, tests,...) 

Finally, create fix tag if everything is fine

```shell
git tag -a 1.3.4 -m "version 1.3.4"
git push origin 1.3.4
```
