# Develop contribution HOWTO

The next instructions describe how to implement new features, resolve bugs, apply
code refactoring or improve the package documentation for biome-text core development team

First of all, start creating an issue with an enough description and assign a label in function of issue nature:

- `bugfix`: when something is wrong
- `feature`: for a new or feature
- `refactor`: for a code refactoring proposal
- `documentation`: for documentation related issues

Once the issue is created, create the local branch following the naming: `<issue_label>`/<`number_of_related_issue`>.

For example, a new issue, #13, describing an error found in documentation, and labelled as `documentation`, will
have an new related branch called `documentation/#13`

Work on this branch make the adequate changes and then push the new branch and create an new PR.

This new PR will include the text "Closes #13" at the end of the description
