# Contributing

Open-Unmix is designed as scientific software. Therefore, we encourage the community to submit bug-fixes and comments and improve the __computational performance__, __reproducibility__ and the __readability__ of the code where possible. When contributing to this repository, please first discuss the change you wish to make in the issue tracker with the owners of this repository before making a change.

We are not looking for contributions that only focus on improving the __separation performance__. However, if this is case, we, instead, encourage researchers to 

1. Use Open-Unmix for their own research, e.g. by modification of the model.
2. Publish and present the results in a scientific paper / conference and __cite open-unmix__.
3. Contact us via mail or open a [performance issue]() if you are interested to contribute the new model.
   In this case we will rerun the training on our internal cluster and update the pre-trained weights accordingly.

Please note we have a code of conduct, please follow it in all your interactions with the project.

## Pull Request Process

The preferred way to contribute to open-unmix is to fork the 
[main repository](http://github.com/sigsep/open-unmix-pytorch/) on
GitHub:

1. Fork the [project repository](http://github.com/sigsep/open-unmix-pytorch):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

```
$ git clone git@github.com:YourLogin/open-unmix-pytorch.git
$ cd open-unmix-pytorch
```

3. Create a branch to hold your changes:

```
$ git checkout -b my-feature
```

   and start making changes. Never work in the ``master`` branch!

4. Ensure any install or build artifacts are removed before making the pull request.

5. Update the README.md and/or the appropriate document in the `/docs` folder with details of changes  to the interface, this includes new command line arguments, dataset description or command line examples.

6. Work on this copy on your computer using Git to do the version
   control. When you're done editing, do:

```
$ git add modified_files
$ git commit
```

   to record your changes in Git, then push them to GitHub with:

```
$ git push -u origin my-feature
```

Finally, go to the web page of your fork of the open-unmix repo,
and click 'Pull request' to send your changes to the maintainers for
review. This will send an email to the committers.

(If any of the above seems like magic to you, then look up the 
[Git documentation](http://git-scm.com/documentation) on the web.)

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as
contributors and maintainers pledge to making participation in our project and
our community a harassment-free experience for everyone, regardless of age, body
size, disability, ethnicity, gender identity and expression, level of experience,
nationality, personal appearance, race, religion, or sexual identity and
orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment
include:

* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery and unwelcome sexual attention or
advances
* Trolling, insulting/derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or electronic
  address, without explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable
behavior and are expected to take appropriate and fair corrective action in
response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or
reject comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct, or to ban temporarily or
permanently any contributor for other behaviors that they deem inappropriate,
threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community. Examples of
representing a project or community include using an official project e-mail
address, posting via an official social media account, or acting as an appointed
representative at an online or offline event. Representation of a project may be
further defined and clarified by project maintainers.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported by contacting the project team @aliutkus, @faroit. All
complaints will be reviewed and investigated and will result in a response that
is deemed necessary and appropriate to the circumstances. The project team is
obligated to maintain confidentiality with regard to the reporter of an incident.
Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good
faith may face temporary or permanent repercussions as determined by other
members of the project's leadership.

### Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage], version 1.4,
available at [http://contributor-covenant.org/version/1/4][version]

[homepage]: http://contributor-covenant.org
[version]: http://contributor-covenant.org/version/1/4/