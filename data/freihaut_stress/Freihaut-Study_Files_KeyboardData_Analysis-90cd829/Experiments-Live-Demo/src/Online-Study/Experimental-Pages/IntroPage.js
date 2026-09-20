import React, { Component } from 'react';

import ProceedButton from '../Layout-Components/ProceedButton';

export default class IntroPage extends Component {

    constructor(props) {
        super(props);
        this.state = {
            checked: false,
        };

        this.toggleCheckBox = this.toggleCheckBox.bind(this);
    }

    componentDidMount() {

        window.scrollTo(0, 0);
    }

    toggleCheckBox() {
        this.state.checked ? this.setState({checked: false}) : this.setState({checked: true})
    }


    render() {
        return(
            <div>
                <div className={"section"}>
                    <div className="content" style={{width: "900px"}}>
                        <div style={{textAlign: "left"}}>
                            <p>
                                Dear participant,
                            </p>
                            <p>
                                Thank you very much for your willingness to participate in this study! This study examines individuals as they complete various computer-based tasks. You will be asked to complete various tasks as well as answer some questions. You will be given detailed instructions about what to do in detail. Please read them carefully and complete all tasks as best you can. The participation time is about 20 to 25 minutes. We will credit you with 10 loyalty points (= 1 EUR) for your participation.
                            </p>
                            <p>
                                Paul Freihaut, M.Sc. <br/>
                                Prof. Dr. Anja Göritz
                            </p>
                            <br/>
                            <div style={{border: "solid 3px red", padding: "5px"}}>
                                <h5 style={{color: "red"}}>
                                    Important technical note
                                </h5>
                                <h6>
                                    Please use a computer mouse and no trackpad, touch or other input devices to participate in the study. Participation on a smartphone or tablet is not possible.
                                </h6>
                            </div>
                            <br/>
                            <h5>
                                Information on participation and the data collected
                            </h5>
                            <p>
                                Your participation is voluntary and you have the right to terminate the study at any time without giving reasons.
                            </p>
                            <p>
                                Questionnaire data and computer mouse and keyboard data are collected in the study. The collection of data is for scientific purposes only. The data collected in the study may be used beyond the scope of this study for secondary analysis or for another scientific question in the interest of scientific transparency and replicability.
                            </p>
                            <p>
                                If you would like more information on how your data will be processed, please click <a target="_blank" rel="noopener noreferrer" href={"https://www.wisopanel.net/datenschutz.php"}>here</a>.
                            </p>
                            <br/>
                            <h5>
                                How to contact us:
                            </h5>
                            <p>
                                Paul Freihaut, M.Sc. <br/>
                                University of Freiburg <br/>
                                Department of Psychology <br/>
                                Occupational and Consumer Psychology <br/>
                                Engelbergerstraße 41 <br/>
                                D-79085 Freiburg <br/>
                                E-Mail: <a href={"mailto:paul.freihaut@psychologie.uni-freiburg.de"}>paul.freihaut@psychologie.uni-freiburg.de</a>
                            </p>
                        <label className="checkbox" style={{display: "flex", marginTop: "45px"}}>
                            <input type="checkbox" style={{marginRight: "15px", flex: "1", outline: "none"}} checked={this.state.checked} onChange={this.toggleCheckBox}/>
                            I have read the information on participantion and agree to participate. I agree to the data collection and further processing
                        </label>
                        </div>
                        <ProceedButton onClick={() => this.props.proceedPhase(this.props.name)} disabled={!this.state.checked}/>
                    </div>
                </div>
            </div>
        );
    }

}