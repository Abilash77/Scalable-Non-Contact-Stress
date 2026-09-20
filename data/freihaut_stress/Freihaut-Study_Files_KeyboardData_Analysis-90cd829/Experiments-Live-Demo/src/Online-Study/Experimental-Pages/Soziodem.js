import React, { Component } from 'react';
import ProceedButton from "../Layout-Components/ProceedButton";

const divStyle = {
    marginTop: "40px",
    textAlign: "left"
};

const questionTextStyle = {
    textAlign: "left",
    fontWeight: "bold"
};


export default class Soziodem extends Component {

    constructor(props) {
        super(props);
        // set a state for each questionnaire item
        this.state = {
            noMouse: -99,
            socioDemographics: {
                mouse: -99,
                mouse_hand: -99,
                keyboard: -99
            }
        };

        this.handleInputChange = this.handleInputChange.bind(this);
        this.isQuestionnaireComplete = this.isQuestionnaireComplete.bind(this);
        this.renderAskIfMouse = this.renderAskIfMouse.bind(this);
        this.renderHasMouse = this.renderHasMouse.bind(this);
        this.renderHasNoMouse = this.renderHasNoMouse.bind(this);
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

        this.setState({socioDemographics})

    }

    evalMouse(mouseState){
        if(mouseState === 0) {
            this.setState({
                noMouse: false
            })
        } else if(mouseState === 1) {
            this.setState({
                noMouse: true
            })
        }
    }

    // ask the partipant wether he is using a mouse
    renderAskIfMouse(){
        return(
            <div>
                <div>
                    <h4>Before we start:</h4>
                    <hr style={{margin: "0 0", height: "3px"}}/>
                </div>

                <div className="field" style={divStyle}>
                    <span>
                        <p style={questionTextStyle}>Are you using a computer mouse right now to process this study? (The trackpad of a laptop is not equivalent to a computer mouse in this study)</p>
                        <div className="control">
                            <label className="radio">
                                <input
                                    style={{marginRight: 5}}
                                    key={0}
                                    type="radio"
                                    name="mouse"
                                    value="0"
                                    onChange={this.handleInputChange}
                                />Yes
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input
                                    style={{marginRight: 5}}
                                    key={1}
                                    type="radio"
                                    name="mouse"
                                    value="1"
                                    onChange={this.handleInputChange}
                                />No
                            </label>
                        </div>
                    </span>
                </div>

                <div style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    marginTop: "58px",
                }} className="control">
                    <button className="button is-link" onClick={() => this.evalMouse(this.state.socioDemographics.mouse)}
                            disabled={this.state.socioDemographics.mouse === -99}>Continue</button>
                </div>
            </div>
        )
    }

    // if the participant uses a mouse, ask with which hand the mouse is operated with
    renderHasMouse(){
        return(
            <div>

                <div>
                    <h4>Before we start:</h4>
                    <hr style={{margin: "0 0", height: "3px"}}/>
                </div>

                <div className="field" style={divStyle}>
                    <span>
                        <p style={questionTextStyle}>With which hand do you operate the computer mouse?</p>
                        <div className="control">
                            <label className="radio">
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    key={2}
                                    name="mouse_hand"
                                    value="0"
                                    onChange={this.handleInputChange}
                                />Right
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    key={3}
                                    name="mouse_hand"
                                    value="1"
                                    onChange={this.handleInputChange}
                                />Left
                            </label>
                        </div>
                    </span>
                </div>

                <div className="field" style={divStyle}>
                    <span>
                        <p style={questionTextStyle}>Are you using a laptop/notebook keyboard right now to work on this study?</p>
                        <div className="control">
                            <label className="radio">
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    key={4}
                                    name="keyboard"
                                    value="0"
                                    onChange={this.handleInputChange}
                                />Yes
                            </label>
                            <label className="radio" style={{marginLeft: 25}}>
                                <input
                                    style={{marginRight: 5}}
                                    type="radio"
                                    key={5}
                                    name="keyboard"
                                    value="1"
                                    onChange={this.handleInputChange}
                                />No
                            </label>
                        </div>
                    </span>
                </div>

                <ProceedButton onClick={() => this.props.proceedPhase(this.props.name, this.state.socioDemographics)}
                               disabled={!this.isQuestionnaireComplete()}/>
            </div>
        )
    }

    // if no mouse is used show info that it is not possible to do the study without a mouse and ask to restart the study
    // with a mouse (or close the study).
    renderHasNoMouse(){

        window.onbeforeunload = null;

        return(
            <div style={{textAlign: "left"}}>
                <h5 style={{color: "red"}}>
                    Important participation notice
                </h5>
                <p>
                    Thank you for informing us that you do not use a computer mouse. However, a computer mouse is necessary for the study to be processed properly.
                </p>
                <p>
                    If you have the possibility to perform the study with a computer mouse (by connecting a mouse or using a computer with a mouse), you can restart the study. If it is not possible for you to process the study with a computer mouse, you can now close this window.
                </p>
                <p>
                    Thank you for your understanding!
                </p>
                <div style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    marginTop: "58px",
                }} className="control">
                    <button className="button is-link" onClick={() => this.props.restart()}
                    >Restart the study</button>
                </div>
            </div>

        )
    }

    render() {

        return(
            <div className="section">
                <div className="content">
                    {this.state.noMouse === -99 ?
                        this.renderAskIfMouse() :
                        this.state.noMouse ?
                            this.renderHasNoMouse() :
                            this.renderHasMouse()}
                </div>
            </div>
        );
    }

}