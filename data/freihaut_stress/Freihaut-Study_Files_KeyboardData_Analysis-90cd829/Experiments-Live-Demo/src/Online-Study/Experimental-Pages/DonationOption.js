import React, { Component } from 'react';

import ProceedButton from '../Layout-Components/ProceedButton';


export default class DonationOption extends Component {

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

    endPhase() {
        const data = {
            isDonating: this.state.checked
        };

        this.props.proceedPhase(this.props.name, data)
    }

    render(){
        return(
            <div>
                <div className={"section"}>
                    <div className="content" style={{width: "900px"}}>
                        <div>
                            <h5>
                                Done!
                            </h5>
                            <h5>
                                For your participation in this study, we will credit you with 10 loyality points (= 1 EUR) after completing the data collection.
                            </h5>
                                <label className="checkbox" style={{marginTop: "45px"}}>
                                    <input type="checkbox" style={{marginRight: "15px", outline: "none"}} checked={this.state.checked} onChange={this.toggleCheckBox}/>
                                    I will waive the loyality points and donate them back to WisoPanel.
                                </label>
                        </div>
                        <ProceedButton onClick={() => this.endPhase()}/>
                    </div>
                </div>
            </div>
        );
    }

}