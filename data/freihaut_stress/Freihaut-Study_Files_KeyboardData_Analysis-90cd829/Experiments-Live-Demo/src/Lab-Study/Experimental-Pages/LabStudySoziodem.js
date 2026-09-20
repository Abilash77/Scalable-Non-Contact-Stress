// Component that shows sociodemografic questions

import React, { Component } from 'react';

import LabStudyProceedButton from '../Layout-Components/LabStudyProceedButton';


// Age, Gender, Education, Computer Fluency, Input Device

const divStyle = {
    marginTop: "40px",
    textAlign: "left"
};

const questionTextStyle = {
    textAlign: "left",
    fontWeight: "bold"
};


export default class LabStudySoziodem extends Component {

    constructor(props) {
        super(props);
        // set a state for each questionnaire item
        this.state = {
            socioDemographics: {
                age: -99,
                sex: -99,
                occupation: -99,
                hand: -99,
            }
        };

        this.handleInputChange = this.handleInputChange.bind(this);
        this.isQuestionnaireComplete = this.isQuestionnaireComplete.bind(this);
    }

    componentDidMount() {
        window.scrollTo(0, 0)
    };

    isQuestionnaireComplete() {
        // check if all questions have a value other than -99 --> each question is answered, if yes, enable proceedphase button
        for (let property in this.state.socioDemographics) {
            if (this.state.socioDemographics.hasOwnProperty(property)) {
                //console.log(property + " = " + this.state[property]);
                if (this.state.socioDemographics[property] === -99) {
                    return false;
                }
            }
        }

        return true;
    }

    handleInputChange(event) {
        // Get the name and value of the clicked radio button and save it to the corresponding question state
        const target = event.target;
        const value = target.value;
        const name = target.name;

        let socioDemographics = {...this.state.socioDemographics};
        socioDemographics[name] = parseInt(value);
        this.setState({socioDemographics});
    }

    render() {

        return(
            <div className="section">
            <div className="content">
                <div>
                    <h4>First, please answer the following questions regarding you person</h4>
                    <hr style={{margin: "0 0", height: "3px"}}/>
                </div>
                <div className="field" style={{marginTop: "25px", textAlign: "left"}}>
                    <p style={questionTextStyle}>Your age:</p>
                    <div className="control">
                        <label>
                            <input
                                style={{width: "150px"}}
                                name="age"
                                className="input"
                                type="text"
                                placeholder="Your age"
                                onChange={this.handleInputChange}
                            />
                            <p style={
                                {color: "hsl(348, 100%, 61%)",
                                    fontSize: "14px",
                                    visibility: isNaN(this.state.socioDemographics.age) ? "visible" : "hidden"
                                }}>Invalid input, input must be a number
                            </p>
                        </label>
                    </div>
                </div>

                <div className="field" style={{marginTop: "26px", textAlign: "left"}}>
                    <span>
                        <p style={questionTextStyle}>Your sex:</p>
                        <div className="control">
                            <label className="radio">
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    value="0"
                                    name="sex"
                                    onChange={this.handleInputChange}
                                />Female
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    value="1"
                                    name="sex"
                                    onChange={this.handleInputChange}
                                />Male
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input style={{marginRight: 5}}
                                       type="radio"
                                       value="2"
                                       name="sex"
                                       onChange={this.handleInputChange}
                                />Third sex
                            </label>
                        </div>
                    </span>
                </div>

                <div className="field" style={divStyle}>
                    <span>
                        <p style={questionTextStyle}>Your profession:</p>
                        <div className="control">
                            <label className="radio">
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    value="0"
                                    name="occupation"
                                    onChange={this.handleInputChange}
                                />In school
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    value="1"
                                    name="occupation"
                                    onChange={this.handleInputChange}
                                />In an apprenticeship
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    value="2"
                                    name="occupation"
                                    onChange={this.handleInputChange}
                                />In college/university
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input style={{marginRight: 5}}
                                       type="radio"
                                       value="3"
                                       name="occupation"
                                       onChange={this.handleInputChange}
                                />Working
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input style={{marginRight: 5}}
                                       type="radio"
                                       value="4"
                                       name="occupation"
                                       onChange={this.handleInputChange}
                                />Other (e.g. retired)
                            </label>
                        </div>
                    </span>
                </div>

                <div className="field" style={divStyle}>
                    <span>
                        <p style={questionTextStyle}>Your dominant hand:</p>
                        <div className="control">
                            <label className="radio">
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    name="hand"
                                    value="0"
                                    onChange={this.handleInputChange}
                                />Right
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    name="hand"
                                    value="1"
                                    onChange={this.handleInputChange}
                                />Left
                            </label>
                        </div>
                    </span>
                </div>

                <LabStudyProceedButton onClick={() => this.props.proceedPhase(this.props.phaseName, this.state.socioDemographics)} disabled={!this.isQuestionnaireComplete()}/>
            </div>
            </div>
        );
    }

}